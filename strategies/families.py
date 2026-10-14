"""
Strategy families — a "genome" (family + parameters) the learning loop can
mutate, cross between families, and promote.

Every family shares the same long-only execution model and risk genes:
  - entry on the NEXT bar's open after an entry event (no same-bar look-ahead)
  - initial stop = entry - atr_mult_initial x ATR
  - profit floors (lock profit_lock_1 of gain after +profit_floor_1, lock
    profit_lock_2 after +profit_floor_2) and ATR trailing that tightens as
    profit grows (atr_mult_initial → atr_mult_tight → atr_mult_tightest)
  - intrabar stop fills at min(open, stop) (gaps are paid, not skipped)
  - family exit signal or max_hold_days → exit next open
  - costs (slippage + spread per side) are charged inside the daily returns

Families differ only in the entry event / exit signal:
  sma_cross       SMA(short) crosses above SMA(long); exit when it crosses back
  momentum        N-day return > threshold and close > SMA(trend); exit when
                  momentum < 0 or close < SMA(trend)
  breakout        close > highest high of prior N bars (optional SMA200
                  filter); exit when close < lowest low of prior M bars
  mean_reversion  RSI(n) < entry level while close > SMA(trend); exit when
                  RSI > exit level (time stop recommended)
  trend_hold      stay invested while close > SMA(n)·(1+buffer); exit below
                  SMA(n)·(1-buffer). High exposure — aims to keep most of
                  buy-and-hold's return while sidestepping long downtrends.

The interface matches what engine.py needs: name, version, min_bars,
sma_short/sma_long (snapshot display), live_signal, initial_stop,
evaluate_exit, backtest.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Optional

import numpy as np
import pandas as pd

from strategies.v1_1 import metrics_from_daily

FAMILIES = ("sma_cross", "momentum", "breakout", "mean_reversion", "trend_hold")

COMMON_DEFAULTS = {
    "atr_period": 14,
    "atr_mult_initial": 3.0,
    "atr_mult_tight": 0.8,
    "atr_mult_tightest": 0.6,
    "profit_floor_1": 0.025,
    "profit_lock_1": 0.50,
    "profit_floor_2": 0.05,
    "profit_lock_2": 0.65,
    "max_hold_days": 0,          # 0 = no time stop
    "cost_per_side": 0.0015,     # slippage 0.10% + half-spread 0.05%
}

FAMILY_DEFAULTS = {
    "sma_cross": {"sma_short": 20, "sma_long": 50},
    "momentum": {"lookback": 63, "threshold_pct": 5.0, "trend_sma": 100},
    "breakout": {"entry_n": 55, "exit_n": 20, "trend_filter": 1},
    "mean_reversion": {"rsi_n": 3, "entry_rsi": 15.0, "exit_rsi": 60.0, "trend_sma": 200,
                       "max_hold_days": 10, "atr_mult_initial": 2.5},
    # wide risk genes: the family's own SMA exit does the work
    "trend_hold": {"sma_n": 200, "buffer_pct": 1.0, "atr_mult_initial": 5.0, "atr_mult_tight": 5.0,
                   "atr_mult_tightest": 4.0, "profit_floor_1": 0.10, "profit_lock_1": 0.3,
                   "profit_floor_2": 0.25, "profit_lock_2": 0.5},
}

# (min, max, kind) — kind: int | float | bool
BOUNDS = {
    "sma_short": (5, 50, "int"), "sma_long": (20, 200, "int"),
    "lookback": (20, 252, "int"), "threshold_pct": (0.0, 30.0, "float"), "trend_sma": (50, 200, "int"),
    "entry_n": (10, 120, "int"), "exit_n": (5, 60, "int"), "trend_filter": (0, 1, "bool"),
    "sma_n": (50, 250, "int"), "buffer_pct": (0.0, 5.0, "float"),
    "rsi_n": (2, 14, "int"), "entry_rsi": (5.0, 40.0, "float"), "exit_rsi": (45.0, 85.0, "float"),
    "atr_mult_initial": (0.5, 6.0, "float"), "atr_mult_tight": (0.3, 6.0, "float"),
    "atr_mult_tightest": (0.3, 6.0, "float"),
    "profit_floor_1": (0.01, 0.10, "float"), "profit_lock_1": (0.2, 0.9, "float"),
    "profit_floor_2": (0.02, 0.25, "float"), "profit_lock_2": (0.3, 0.9, "float"),
    "max_hold_days": (0, 120, "int"),
}

FAMILY_GENES = {
    "sma_cross": ("sma_short", "sma_long"),
    "momentum": ("lookback", "threshold_pct", "trend_sma"),
    "breakout": ("entry_n", "exit_n", "trend_filter"),
    "mean_reversion": ("rsi_n", "entry_rsi", "exit_rsi", "trend_sma"),
    "trend_hold": ("sma_n", "buffer_pct"),
}
RISK_GENES = ("atr_mult_initial", "atr_mult_tight", "atr_mult_tightest", "profit_floor_1",
              "profit_lock_1", "profit_floor_2", "profit_lock_2", "max_hold_days")


# ---------------------------------------------------------------------- #
# Genome helpers
# ---------------------------------------------------------------------- #

def normalize(genome: dict) -> dict:
    """Fill defaults; drop keys that don't belong to the family."""
    fam = genome.get("family", "sma_cross")
    if fam not in FAMILIES:
        raise ValueError(f"unknown strategy family {fam!r}")
    g = {"family": fam, **COMMON_DEFAULTS, **FAMILY_DEFAULTS[fam]}
    for k, v in genome.items():
        if k in g or k in ("version",):
            g[k] = v
    for k, (lo, hi, kind) in BOUNDS.items():
        if k in g:
            g[k] = int(round(g[k])) if kind in ("int", "bool") else float(g[k])
    return g


def genome_hash(genome: dict) -> str:
    g = {k: v for k, v in normalize(genome).items() if k != "version"}
    return hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()[:8]


def validate(genome: dict) -> list:
    """Bounds + structural sanity. Empty list = valid."""
    try:
        g = normalize(genome)
    except ValueError as e:
        return [str(e)]
    errs = []
    for k, (lo, hi, _) in BOUNDS.items():
        if k in g and not (lo <= g[k] <= hi):
            errs.append(f"{k}={g[k]} outside [{lo}, {hi}]")
    if g["family"] == "sma_cross" and g["sma_short"] >= g["sma_long"]:
        errs.append("sma_short must be < sma_long")
    if g["profit_floor_1"] >= g["profit_floor_2"]:
        errs.append("profit_floor_1 must be < profit_floor_2")
    if g["family"] == "mean_reversion" and g["entry_rsi"] >= g["exit_rsi"]:
        errs.append("entry_rsi must be < exit_rsi")
    return errs


def mutate(genome: dict, rng: random.Random, n_genes: int = 2) -> dict:
    """Perturb up to n_genes family/risk genes by ±10–35% (or flip a bool)."""
    g = normalize(genome)
    keys = list(FAMILY_GENES[g["family"]]) + list(RISK_GENES)
    for k in rng.sample(keys, k=min(n_genes, len(keys))):
        lo, hi, kind = BOUNDS[k]
        if kind == "bool":
            g[k] = 1 - g[k]
            continue
        base = g[k] if g[k] else (lo + hi) / 4
        v = base * rng.choice([0.65, 0.8, 0.9, 1.1, 1.25, 1.35])
        v = min(hi, max(lo, v))
        g[k] = int(round(v)) if kind == "int" else round(v, 3)
    g.pop("version", None)
    return g


def random_genome(family: str, rng: random.Random) -> dict:
    """A fresh genome in `family` — used for exploration across categories."""
    g = normalize({"family": family})
    for k in FAMILY_GENES[family]:
        lo, hi, kind = BOUNDS[k]
        if kind == "bool":
            g[k] = rng.choice([0, 1])
        elif kind == "int":
            g[k] = rng.randint(int(lo), int(hi))
        else:
            g[k] = round(rng.uniform(lo, hi), 2)
    if family == "sma_cross" and g["sma_short"] >= g["sma_long"]:
        g["sma_short"], g["sma_long"] = sorted((max(5, g["sma_long"] // 3), g["sma_long"]))
    if family == "mean_reversion" and g["entry_rsi"] >= g["exit_rsi"]:
        g["entry_rsi"], g["exit_rsi"] = 15.0, 60.0
    return g


def describe(genome: dict) -> str:
    g = normalize(genome)
    fam = ", ".join(f"{k}={g[k]}" for k in FAMILY_GENES[g["family"]])
    return f"{g['family']}({fam}; stop={g['atr_mult_initial']}xATR, hold≤{g['max_hold_days'] or '∞'})"


# ---------------------------------------------------------------------- #
# Indicators
# ---------------------------------------------------------------------- #

def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    c = df["Close"]
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - c.shift()).abs(),
                    (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


# ---------------------------------------------------------------------- #
# Strategy
# ---------------------------------------------------------------------- #

class GenomeStrategy:
    def __init__(self, genome: dict):
        self.genome = normalize(genome)
        g = self.genome
        self.family = g["family"]
        for k, v in g.items():
            if k != "family":
                setattr(self, k, v)
        # snapshot display windows (SMA values are shown to the models)
        self.sma_short = g.get("sma_short", 20)
        self.sma_long = g.get("sma_long", 50)
        self.version = genome.get("version") or f"{self.family}-{genome_hash(g)}"
        self.name = describe(g)

    # --- signals --------------------------------------------------------
    @property
    def min_bars(self) -> int:
        g = self.genome
        need = {
            "sma_cross": g.get("sma_long", 50),
            "momentum": max(g.get("lookback", 63), g.get("trend_sma", 100)),
            "breakout": max(g.get("entry_n", 55), 200 if g.get("trend_filter") else 0),
            "mean_reversion": g.get("trend_sma", 200),
            "trend_hold": g.get("sma_n", 200),
        }[self.family]
        return int(max(need, self.atr_period) + 2)

    def _signals(self, df: pd.DataFrame):
        """(entry_event, exit_signal) as boolean numpy arrays, evaluated at bar close."""
        g, c = self.genome, df["Close"]
        if self.family == "sma_cross":
            s = c.rolling(g["sma_short"]).mean() > c.rolling(g["sma_long"]).mean()
            valid = c.rolling(g["sma_long"]).mean().notna()
            entry = s & ~s.shift(1, fill_value=False) & valid
            exit_ = ~s & valid
        elif self.family == "momentum":
            mom = c / c.shift(g["lookback"]) - 1
            trend = c.rolling(g["trend_sma"]).mean()
            cond = (mom * 100 > g["threshold_pct"]) & (c > trend)
            entry = cond & ~cond.shift(1, fill_value=False)
            exit_ = ((mom < 0) | (c < trend)) & trend.notna()
        elif self.family == "breakout":
            hh = df["High"].rolling(g["entry_n"]).max().shift(1)
            ll = df["Low"].rolling(g["exit_n"]).min().shift(1)
            entry = c > hh
            if g["trend_filter"]:
                entry = entry & (c > c.rolling(200).mean())
            exit_ = c < ll
        elif self.family == "trend_hold":
            sma = c.rolling(g["sma_n"]).mean()
            above = c > sma * (1 + g["buffer_pct"] / 100)
            entry = above & ~above.shift(1, fill_value=False) & sma.notna()
            exit_ = (c < sma * (1 - g["buffer_pct"] / 100)) & sma.notna()
        else:  # mean_reversion
            rsi = _rsi(c, g["rsi_n"])
            trend = c.rolling(g["trend_sma"]).mean()
            entry = (rsi < g["entry_rsi"]) & (c > trend)
            exit_ = rsi > g["exit_rsi"]
        return entry.fillna(False).to_numpy(bool), exit_.fillna(False).to_numpy(bool)

    def live_signal(self, df: pd.DataFrame, entry_lookback_bars: int = 3) -> int:
        """1 = entry event within the last N bars and no exit signal now;
        -1 = exit signal now; 0 = otherwise."""
        if len(df) < self.min_bars:
            return 0
        entry, exit_ = self._signals(df)
        if exit_[-1]:
            return -1
        if entry[-entry_lookback_bars:].any():
            return 1
        return 0

    def initial_stop(self, df: pd.DataFrame) -> Optional[float]:
        atr = _atr(df, self.atr_period).iloc[-1]
        if pd.isna(atr) or atr <= 0:
            return None
        return float(df["Close"].iloc[-1] - self.atr_mult_initial * atr)

    def _ratchet(self, entry: float, stop: float, close: float, atr: float) -> tuple:
        gain = (close - entry) / entry if entry > 0 else 0.0
        if gain >= self.profit_floor_2:
            stop = max(stop, entry * (1 + gain * self.profit_lock_2))
            mult = self.atr_mult_tightest
        elif gain >= self.profit_floor_1:
            stop = max(stop, entry * (1 + gain * self.profit_lock_1))
            mult = self.atr_mult_tight
        else:
            mult = self.atr_mult_initial
        if atr > 0:
            stop = max(stop, close - mult * atr)
        return stop, mult

    def evaluate_exit(self, df: pd.DataFrame, entry_price: float,
                      current_stop: Optional[float], bars_held: Optional[int] = None) -> dict:
        atr_s = _atr(df, self.atr_period)
        atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else 0.0
        close = float(df["Close"].iloc[-1])
        stop = float(current_stop) if current_stop else (
            entry_price - self.atr_mult_initial * atr if atr > 0 else entry_price * 0.9)
        _, exit_ = self._signals(df)
        if exit_[-1]:
            return {"exit": True, "reason": f"{self.family}_exit_signal", "stop": stop}
        if self.max_hold_days and bars_held is not None and bars_held >= self.max_hold_days:
            return {"exit": True, "reason": f"time_stop_{bars_held}d", "stop": stop}
        stop, mult = self._ratchet(entry_price, stop, close, atr)
        if close <= stop:
            return {"exit": True, "reason": f"stop_hit_{mult}x_atr", "stop": stop}
        return {"exit": False, "reason": "", "stop": stop}

    # --- backtest -------------------------------------------------------
    def backtest(self, df: pd.DataFrame) -> dict:
        n = len(df)
        empty = {"total_return_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0,
                 "total_trades": 0, "win_rate": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
                 "total_wins": 0, "total_losses": 0, "trade_returns": [], "trades": [],
                 "exit_reasons": {}, "costs_included": True,
                 "daily_returns": pd.Series(np.zeros(n), index=df.index)}
        if n < self.min_bars + 2:
            empty["error"] = "insufficient data"
            return empty

        o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
        atr = _atr(df, self.atr_period).to_numpy(float)
        entry_ev, exit_sig = self._signals(df)
        cost = self.cost_per_side
        idx = df.index

        daily = np.zeros(n)
        trades = []
        pos = pending_entry = pending_exit = False
        entry_px = stop = 0.0
        entry_i = held = 0
        reason = ""

        def close_trade(i, px, why):
            trades.append({"entry_date": idx[entry_i], "exit_date": idx[i],
                           "pnl_pct": ((px * (1 - cost)) / (entry_px * (1 + cost)) - 1) * 100,
                           "exit_reason": why})

        for i in range(1, n):
            if pos:
                if pending_exit:
                    daily[i] = o[i] / c[i - 1] - 1 - cost
                    close_trade(i, o[i], reason)
                    pos = pending_exit = False
                elif l[i] <= stop:
                    px = min(o[i], stop)
                    daily[i] = px / c[i - 1] - 1 - cost
                    close_trade(i, px, "stop")
                    pos = False
                else:
                    daily[i] = c[i] / c[i - 1] - 1
                    held += 1
                    stop, _ = self._ratchet(entry_px, stop, c[i], atr[i] if atr[i] == atr[i] else 0.0)
                    if exit_sig[i]:
                        pending_exit, reason = True, f"{self.family}_exit_signal"
                    elif self.max_hold_days and held >= self.max_hold_days:
                        pending_exit, reason = True, "time_stop"
            if not pos and pending_entry:
                pending_entry = False
                a = atr[i - 1]
                if a == a and a > 0:
                    entry_px, entry_i, held = o[i], i, 0
                    stop = entry_px - self.atr_mult_initial * a
                    pos = True
                    if l[i] <= stop:  # stopped out on the entry bar
                        daily[i] += stop / o[i] - 1 - 2 * cost
                        close_trade(i, stop, "stop")
                        pos = False
                    else:
                        daily[i] += c[i] / o[i] - 1 - cost
                        stop, _ = self._ratchet(entry_px, stop, c[i], a)
                        if exit_sig[i]:
                            pending_exit, reason = True, f"{self.family}_exit_signal"
            if not pos and not pending_entry and entry_ev[i] and i < n - 1:
                pending_entry = True

        if pos:
            close_trade(n - 1, c[-1], "end_of_data")

        daily_s = pd.Series(daily, index=idx)
        m = metrics_from_daily(daily_s.iloc[1:])
        rets = [t["pnl_pct"] for t in trades]
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]
        reasons = {}
        for t in trades:
            reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1
        return {
            **m,
            "total_trades": len(trades),
            "win_rate": round(len(wins) / len(rets) * 100, 1) if rets else 0.0,
            "avg_win_pct": round(float(np.mean(wins)), 2) if wins else 0.0,
            "avg_loss_pct": round(float(np.mean(losses)), 2) if losses else 0.0,
            "total_wins": len(wins),
            "total_losses": len(losses),
            "trade_returns": rets,
            "trades": trades,
            "exit_reasons": reasons,
            "daily_returns": daily_s,
            "costs_included": True,
        }


def make_strategy(genome: Optional[dict] = None) -> GenomeStrategy:
    return GenomeStrategy(genome or {"family": "sma_cross"})


DEFAULT_GENOME = {"family": "sma_cross", **FAMILY_DEFAULTS["sma_cross"], "version": "v1.1"}
