"""
ETF trend strategy (v2.0) — diversified, monthly, long-only, no leverage.

Rule (Faber 2007 / Hurst-Ooi-Pedersen 2017, tested in
scripts/etf_trend_backtest.py):

  - 9 asset-class ETFs. Each is held while its price is above its 10-month
    average, with a ±1% buffer: buy only above avg x 1.01, sell only below
    avg x 0.99 (stops flip-flopping around the line).
  - Weights are inverse-volatility (63-day), so a calm bond fund and a
    volatile emerging-market fund carry similar risk. Out-of-market slots
    go to T-bills (BIL).
  - The portfolio is split into 4 tranches that rebalance on trading days
    1, 6, 11 and 16 of each month. Each tranche samples its "monthly" prices
    on its own day; the target is the average of the 4 tranches. This removes
    the luck of one rebalance day (worst drawdown ranged −10% to −20% by day).

Everything is a pure function of daily closes up to the decision date, so the
live bot, the replay and the backtest compute the same targets. Parameters are
fixed by design: re-tuning them yearly did worse in walk-forward tests.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

VERSION = "etf_trend_v2.0"
DEFAULTS = {
    "symbols": ["SPY", "IWM", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "VNQ"],
    "cash_symbol": "BIL",
    "lookback_months": 10,
    "band": 0.01,
    "vol_window": 63,
    "min_vol": 0.02,
    "max_weight": 0.35,
    "weighting": "invvol",          # invvol | equal
    "tranche_days": [0, 5, 10, 15],  # 0-based trading day of the month
}


def params_from_config(config: dict) -> dict:
    return {**DEFAULTS, **(config.get("etf_trend") or {})}


def _naive_dates(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def day_of_month(index) -> pd.Series:
    """0-based trading-day number within each calendar month."""
    idx = _naive_dates(index)
    s = pd.Series(1, index=idx)
    return s.groupby(idx.to_period("M")).cumsum() - 1


class TrendBook:
    """Tranche states for one price history. Call rebalance(d) on each tranche
    day in order; combined() gives the current target weights."""

    def __init__(self, closes: pd.DataFrame, params: Optional[dict] = None):
        self.p = {**DEFAULTS, **(params or {})}
        closes = closes.copy()
        closes.index = _naive_dates(closes.index)
        self.closes = closes[[s for s in self.p["symbols"] if s in closes.columns]]
        self.dom = day_of_month(self.closes.index)
        rets = self.closes.pct_change()
        self.vol = rets.rolling(self.p["vol_window"]).std() * np.sqrt(252)
        L = int(self.p["lookback_months"])
        self.samples, self.sma, self.count = {}, {}, {}
        for k in self.p["tranche_days"]:
            m = self.closes[self.dom == k]
            self.samples[k] = m
            self.sma[k] = m.rolling(L, min_periods=L).mean()
        cols = list(self.closes.columns)
        self.state = {k: dict.fromkeys(cols, 0) for k in self.p["tranche_days"]}
        self.weights = {k: None for k in self.p["tranche_days"]}
        self.signals = {k: {} for k in self.p["tranche_days"]}
        self.last = {k: None for k in self.p["tranche_days"]}

    def rebalance(self, d) -> list:
        """Update every tranche whose rebalance day is d. Returns tranches updated."""
        d = pd.Timestamp(d).normalize()
        if d not in self.dom.index:
            return []
        k_today = int(self.dom.loc[d])
        done = []
        for k in self.p["tranche_days"]:
            if k != k_today:
                continue
            m, sma = self.samples[k], self.sma[k]
            band = float(self.p["band"])
            avail = [c for c in self.closes.columns
                     if pd.notna(sma.at[d, c]) and pd.notna(m.at[d, c])]
            sig = {}
            for c in avail:
                p, a = float(m.at[d, c]), float(sma.at[d, c])
                st = self.state[k][c]
                if st == 0 and p > a * (1 + band):
                    st = 1
                elif st == 1 and p < a * (1 - band):
                    st = 0
                self.state[k][c] = st
                sig[c] = {"price": p, "avg": a, "pct_vs_avg": (p / a - 1) * 100, "in": bool(st)}
            self.weights[k] = self._weights(d, avail, k)
            self.signals[k] = sig
            self.last[k] = d
            done.append(k)
        return done

    def _weights(self, d, avail: list, k) -> pd.Series:
        w = pd.Series(0.0, index=self.closes.columns)
        if not avail:
            return w
        if self.p["weighting"] == "invvol":
            v = self.vol.loc[d, avail].fillna(self.vol.loc[:d, avail].mean()).clip(lower=self.p["min_vol"])
            base = (1 / v) / (1 / v).sum()
        else:
            base = pd.Series(1 / len(avail), index=avail)
        base = base.clip(upper=float(self.p["max_weight"]))
        for c in avail:
            w[c] = float(base[c]) * self.state[k][c]
        return w

    def combined(self) -> pd.Series:
        """Average of the tranche targets (tranches not started yet count as cash)
        plus the cash symbol, summing to 1."""
        ks = self.p["tranche_days"]
        tot = pd.Series(0.0, index=self.closes.columns)
        for k in ks:
            if self.weights[k] is not None:
                tot = tot + self.weights[k]
        tot = tot / len(ks)
        out = tot[tot > 1e-9]
        out[self.p["cash_symbol"]] = max(0.0, 1.0 - float(out.sum()))
        return out


def targets_asof(closes: pd.DataFrame, asof=None, params: Optional[dict] = None) -> tuple:
    """Target weights using only closes up to `asof` (default: last row).
    Returns (weights incl. cash symbol, info) — the single source of truth for
    the live bot, replay and backtest."""
    p = {**DEFAULTS, **(params or {})}
    c = closes.copy()
    c.index = _naive_dates(c.index)
    if asof is not None:
        c = c[c.index <= pd.Timestamp(asof).normalize()]
    book = TrendBook(c, p)
    days = set(p["tranche_days"])
    for d, k in book.dom.items():
        if int(k) in days:
            book.rebalance(d)
    d = c.index[-1] if len(c) else None
    info = {
        "asof": None if d is None else str(d.date()),
        "rebalance_today": bool(d is not None and int(book.dom.iloc[-1]) in days),
        "tranche_last": {str(k): (None if v is None else str(v.date())) for k, v in book.last.items()},
        "signals": {str(k): v for k, v in book.signals.items()},
        "in_market": {s: sum(book.state[k].get(s, 0) for k in p["tranche_days"])
                      for s in book.closes.columns},
        "tranches": len(p["tranche_days"]),
    }
    return book.combined(), info


def is_rebalance_day(dates, d=None, params: Optional[dict] = None) -> bool:
    p = {**DEFAULTS, **(params or {})}
    dom = day_of_month(dates)
    d = dom.index[-1] if d is None else pd.Timestamp(d).normalize()
    return d in dom.index and int(dom.loc[d]) in set(p["tranche_days"])


def plan_orders(targets: pd.Series, holdings: dict, prices: dict, equity: float, cash: float,
                min_trade_pct: float = 0.5, cash_buffer_pct: float = 1.0, borrow_limit: float = 0.0) -> list:
    """
    Orders that move the account to `targets` (weights of equity).

    holdings: symbol -> shares currently held (any symbol; ones without a
              target are sold completely). prices: symbol -> price.
    Sells first, then buys scaled to fit cash + sell proceeds, keeping a cash
    buffer. Whole shares only. Trades smaller than min_trade_pct of equity
    are skipped (except full exits). borrow_limit (share of equity, default 0)
    lets buys use margin; only the index/leverage overlay sets it.
    """
    if equity <= 0:
        return []
    invest = equity * (1 - cash_buffer_pct / 100)
    floor_ = equity * min_trade_pct / 100
    sells, buys = [], []
    for s in sorted(set(holdings) | set(targets.index)):
        px = float(prices.get(s) or 0)
        held = float(holdings.get(s, 0) or 0)
        want_w = float(targets.get(s, 0.0))
        if px <= 0:
            continue
        cur = held * px
        want = want_w * invest
        diff = want - cur
        if want_w <= 0 and held > 0:
            sells.append({"symbol": s, "side": "sell", "qty": held, "price": px, "reason": "exit"})
        elif diff <= -floor_:
            q = min(held, np.floor(-diff / px))
            if q > 0:
                sells.append({"symbol": s, "side": "sell", "qty": float(q), "price": px, "reason": "trim"})
        elif diff >= floor_:
            q = np.floor(diff / px)
            if q > 0:
                buys.append({"symbol": s, "side": "buy", "qty": float(q), "price": px,
                             "reason": "enter" if held <= 0 else "add"})
    budget = (cash + sum(o["qty"] * o["price"] for o in sells) - equity * cash_buffer_pct / 100
              + equity * max(0.0, borrow_limit))
    need = sum(o["qty"] * o["price"] for o in buys)
    if buys and need > budget:
        scale = max(0.0, budget / need)
        for o in buys:
            o["qty"] = float(np.floor(o["qty"] * scale))
        buys = [o for o in buys if o["qty"] > 0]
    return sells + buys


def simulate(close: pd.DataFrame, rf: pd.Series, target, rdates, open_: Optional[pd.DataFrame] = None,
             cost: float = 0.001, start=None, once: bool = False, borrow_rate: Optional[float] = None) -> tuple:
    """
    Daily account model used by the backtest and the paper-tracking monitor.

    target(d) -> weights (Series) decided on the close of a rebalance day d;
    filled the NEXT day at the open (if open_ given) or close. Holdings drift
    between rebalances, costs are charged on traded notional, uninvested cash
    earns rf. Negative cash (margin, overlay only) pays borrow_rate a year
    when given, else rf. Returns (equity Series starting at 1.0, turnover per year).
    """
    cols = list(close.columns)
    C = close.to_numpy(float)
    O = open_.reindex(index=close.index, columns=cols).to_numpy(float) if open_ is not None else None
    R = rf.reindex(close.index).fillna(0.0).to_numpy(float)
    dates = close.index
    i0 = dates.get_loc(start) if start is not None else 0
    rset = set(rdates)
    h = np.zeros(len(cols))
    cash = 1.0
    eq = np.full(len(dates), np.nan)
    eq[i0] = 1.0
    pending, traded = None, 0.0
    b_day = None if borrow_rate is None else borrow_rate / 252

    def accrue(c, r):
        return c * (1 + (b_day if (b_day is not None and c < 0) else r))

    def trade(target_w, E):
        nonlocal h, cash, traded
        tgt = np.nan_to_num(target_w) * E
        t = np.abs(tgt - h).sum()
        traded += t / E
        h = tgt
        cash = E - h.sum() - t * cost

    if dates[i0] in rset or once:
        pending = target(dates[i0]).reindex(cols).to_numpy(float)
    for i in range(i0 + 1, len(dates)):
        rc = np.nan_to_num(C[i] / C[i - 1] - 1)
        if pending is not None and O is not None:
            ok = np.isfinite(O[i]) & np.isfinite(C[i - 1])
            ov = np.nan_to_num(np.where(ok, O[i] / np.where(ok, C[i - 1], 1) - 1, rc))  # no open → fill at close
            it = np.nan_to_num(np.where(ok, C[i] / np.where(ok, O[i], 1) - 1, 0.0))
            h = h * (1 + ov)
            trade(pending, h.sum() + cash)
            h = h * (1 + it)
            cash = accrue(cash, R[i])
            pending = None
        else:
            h = h * (1 + rc)
            cash = accrue(cash, R[i])
            if pending is not None:
                trade(pending, h.sum() + cash)
                pending = None
        eq[i] = h.sum() + cash
        if dates[i] in rset and not once:
            pending = target(dates[i]).reindex(cols).to_numpy(float)
    e = pd.Series(eq, index=dates).dropna()
    return e, traded / max(len(e) / 252, 1e-9)


def model_equity(closes: pd.DataFrame, opens: Optional[pd.DataFrame], start, params: Optional[dict] = None,
                 cost: float = 0.001) -> pd.Series:
    """What the strategy 'should' have done from `start`: tranche states warmed
    up on prior history, cash symbol held as an asset (as the live bot does).
    The index/leverage overlay (params['overlay'], from config) is applied when enabled."""
    from strategies import overlay as ov
    p = {**DEFAULTS, **(params or {})}
    c = closes.copy()
    c.index = _naive_dates(c.index)
    o = None
    if opens is not None:
        o = opens.copy()
        o.index = _naive_dates(o.index)
    start = pd.Timestamp(start).normalize()
    start = c.index[c.index.searchsorted(start)]
    book = TrendBook(c, p)
    dom = day_of_month(c.index)
    rdt = c.index[dom.isin(p["tranche_days"]).to_numpy()]
    for d in rdt[rdt < start]:
        book.rebalance(d)

    ovs = ov.settings(p)

    def target(d):
        book.rebalance(d)
        return ov.apply(book.combined(), ovs, p["cash_symbol"])
    e, _ = simulate(c, pd.Series(0.0, index=c.index), target, set(rdt) | {start}, o, cost=cost, start=start,
                    borrow_rate=ovs["borrow_rate_pct"] / 100 if ov.active(ovs) else None)
    return e
