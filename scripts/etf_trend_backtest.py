#!/usr/bin/env python3
"""
Detailed backtest of the proposed ETF trend design vs SPY and 60/40.

Realism (vs the quick checks in chat):
  - account simulation: holdings drift daily, rebalanced to target on signal days
  - decisions on a close, executed NEXT day: at the open (ETF era) or the
    close (extended era, where proxies are mutual funds priced once a day)
  - costs on traded notional (default 0.10%/side), cash earns the 13-week
    T-bill rate (^IRX), no leverage, no look-ahead (signal uses closes <= t)

Tests:
  1. ETF era (2007-2026, real ETFs, next-open fills)
  2. Extended era (1997-2026, index-fund proxies spliced before each ETF's
     launch; adds the 2000-02 crash)
  3. Parameter sensitivity (lookback x buffer), signal type, weighting
  4. Rebalance-day "timing luck" (every trading day of the month)
  5. Walk-forward: each year use the parameters that were best on prior data
  6. Statistics: PSR, deflated Sharpe over every variant tried, block-bootstrap
     probability that the strategy's Sharpe beats SPY / 60-40
  7. Crash windows, calendar years, cost stress (0.25%/side)

Usage: .venv/bin/python scripts/etf_trend_backtest.py   (writes reports/etf_trend/)
"""
from __future__ import annotations

import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BOT_DIR)
from learning import stats  # noqa: E402

OUT = os.path.join(BOT_DIR, "reports", "etf_trend")

# sleeve, ETF, pre-launch proxy (daily NAV / index)
SLEEVES = [
    ("US stocks", "SPY", "VFINX"),
    ("US small caps", "IWM", "NAESX"),
    ("Intl developed", "EFA", "VGTSX"),
    ("Emerging markets", "EEM", "VEIEX"),
    ("Treasuries 7-10y", "IEF", "VFITX"),
    ("Treasuries 20y+", "TLT", "VUSTX"),
    ("Gold", "GLD", "GC=F"),
    ("Commodities", "DBC", "^SPGSCI"),
    ("Real estate", "VNQ", "VGSIX"),
]
ETFS = [e for _, e, _ in SLEEVES]
COST = 0.0010
BASE = dict(signal="sma", lookback=10, band=0.01, weighting="equal")
FINAL = "Final design (4 tranches, inv-vol)"
CRASHES = [("Dot-com bust", "2000-03-24", "2002-10-09"), ("2008 crisis", "2007-10-09", "2009-03-09"),
           ("COVID crash", "2020-02-19", "2020-03-23"), ("2022 stocks+bonds", "2022-01-03", "2022-10-12"),
           ("2009-20 bull", "2009-03-09", "2020-02-19"), ("2023-26 bull", "2022-10-12", "2026-12-31")]


# ------------------------------------------------------------------ data
def load():
    import yfinance as yf
    tick = sorted(set(ETFS + [p for _, _, p in SLEEVES] + ["^IRX"]))
    raw = yf.download(tick, period="max", auto_adjust=True, progress=False, group_by="ticker", threads=True)
    cal = raw["SPY"]["Close"].dropna().index
    get = lambda t, f: raw[t][f].reindex(cal)
    irx = get("^IRX", "Close").ffill().fillna(0.0)
    rf = (irx / 100.0 / 252.0).rename("rf")

    etf_close = pd.DataFrame({e: get(e, "Close") for e in ETFS})
    etf_open = pd.DataFrame({e: get(e, "Open") for e in ETFS})

    spliced = {}
    for _, e, p in SLEEVES:
        er = etf_close[e].pct_change()
        pr = get(p, "Close").ffill(limit=5).pct_change()
        first = etf_close[e].first_valid_index()
        r = pr.where(pr.index < first, er)
        start = r.first_valid_index()
        idx = (1 + r.fillna(0.0)).cumprod()
        spliced[e] = idx.where(idx.index >= start)
    ext_close = pd.DataFrame(spliced)
    # drop the final row if any ETF close is missing (partial current session)
    while etf_close.iloc[-1].isna().any():
        etf_close, etf_open, ext_close, rf = etf_close.iloc[:-1], etf_open.iloc[:-1], ext_close.iloc[:-1], rf.iloc[:-1]
    return etf_close, etf_open, ext_close, rf


# ------------------------------------------------------------------ engine
def rebalance_dates(idx: pd.DatetimeIndex, day: int = -1) -> pd.DatetimeIndex:
    """day=-1: last trading day of each month; day=k: k-th trading day (0-based)."""
    g = pd.Series(idx, index=idx).groupby(idx.to_period("M"))
    out = g.nth(day) if day >= 0 else g.last()
    return pd.DatetimeIndex(out.dropna().values)


class TrendSignal:
    def __init__(self, close, rf, rdates, signal="sma", lookback=10, band=0.01, weighting="equal", **_):
        self.close, self.signal, self.L, self.band, self.weighting = close, signal, lookback, band, weighting
        self.m = close.reindex(rdates)                      # monthly samples on rebalance days
        self.sma = self.m.rolling(lookback).mean()
        self.mom = self.m / self.m.shift(lookback) - 1
        rf_cum = (1 + rf).cumprod().reindex(rdates)
        self.rf_mom = rf_cum / rf_cum.shift(lookback) - 1
        self.vol = close.pct_change().rolling(63).std() * np.sqrt(252)
        self.state = {c: 0 for c in close.columns}
        self.flips = 0

    def __call__(self, d) -> pd.Series:
        avail = [c for c in self.close.columns
                 if self.m.loc[:d, c].notna().sum() > self.L and pd.notna(self.close.at[d, c])]
        for c in avail:
            p = self.m.at[d, c]
            if self.signal == "sma":
                up, down = p > self.sma.at[d, c] * (1 + self.band), p < self.sma.at[d, c] * (1 - self.band)
            else:  # absolute momentum: lookback-month return vs T-bills
                ex = self.mom.at[d, c] - self.rf_mom.at[d]
                up, down = ex > self.band, ex < -self.band
            new = 1 if (self.state[c] == 0 and up) else 0 if (self.state[c] == 1 and down) else self.state[c]
            self.flips += new != self.state[c]
            self.state[c] = new
        w = pd.Series(0.0, index=self.close.columns)
        if not avail:
            return w
        if self.weighting == "invvol":
            iv = 1 / self.vol.loc[d, avail].clip(lower=0.02)
            base = iv / iv.sum()
        else:
            base = pd.Series(1 / len(avail), index=avail)
        for c in avail:
            w[c] = base[c] * self.state[c]
        return w


class Fixed:
    def __init__(self, weights: dict):
        self.w, self.flips = weights, 0

    def __call__(self, d) -> pd.Series:
        return pd.Series(self.w)


from strategies.etf_trend import simulate  # noqa: E402  (shared with the live monitor)


# ------------------------------------------------------------------ metrics
def metrics(e: pd.Series, rf: pd.Series) -> dict:
    r = e.pct_change().dropna()
    ex = r - rf.reindex(r.index).fillna(0.0)
    yrs = len(r) / 252
    dd = e / e.cummax() - 1
    under, longest = 0, 0
    for v in (dd < 0).to_numpy():
        under = under + 1 if v else 0
        longest = max(longest, under)

    return {"CAGR %": (e.iloc[-1] ** (1 / yrs) - 1) * 100, "Vol %": r.std() * np.sqrt(252) * 100,
            "Sharpe": ex.mean() / r.std() * np.sqrt(252), "Max DD %": dd.min() * 100,
            "Longest underwater (yrs)": longest / 252,
            "Worst 12m %": (e / e.shift(252) - 1).min() * 100,
            "$100k becomes": 100000 * e.iloc[-1] / e.iloc[0]}


def stationary_bootstrap(X: np.ndarray, n: int = 2000, block: int = 21, seed: int = 7) -> np.ndarray:
    """Politis-Romano stationary bootstrap of rows of X (T x k). Returns (n, k) annualised Sharpes."""
    rng = np.random.default_rng(seed)
    T = len(X); p = 1 / block
    out = np.empty((n, X.shape[1]))
    for b in range(n):
        idx = np.empty(T, dtype=int)
        idx[0] = rng.integers(T)
        jump = rng.random(T) < p
        rnd = rng.integers(T, size=T)
        for t in range(1, T):
            idx[t] = rnd[t] if jump[t] else (idx[t - 1] + 1) % T
        s = X[idx]
        out[b] = s.mean(0) / s.std(0, ddof=1) * np.sqrt(252)
    return out


def fmt(df: pd.DataFrame, nd=2) -> str:
    return df.round(nd).to_markdown()


# ------------------------------------------------------------------ main
def run_era(name, close, rf, open_=None, lines=None, deep=True, core=("SPY", "IEF")):
    rd = rebalance_dates(close.index)
    # common start: first rebalance day with 13 monthly samples of the core sleeves
    core = close[list(core)].reindex(rd).dropna()
    start = core.index[12]
    start = close.index[close.index.get_loc(start)]
    ev = lambda e: e[e.index >= start]

    def strat(**kw):
        p = {**BASE, **kw}
        sig = TrendSignal(close, rf, rd, **p)
        e, to = simulate(close, rf, sig, rd, open_, cost=p.get("cost", COST), start=start)
        return e, to, sig.flips / (len(e) / 252)

    res, curves = {}, {}
    e, to, fl = strat(); curves["Trend (10-mo, 1% buffer)"] = e
    res["Trend (10-mo, 1% buffer)"] = {**metrics(e, rf), "Switches/yr": fl, "Turnover/yr": to}
    e2, to2, fl2 = strat(weighting="invvol")
    res["Trend + inverse-vol weights"] = {**metrics(e2, rf), "Switches/yr": fl2, "Turnover/yr": to2}
    # Final design = the live strategy module itself (strategies/etf_trend.py)
    from strategies.etf_trend import TrendBook, day_of_month, DEFAULTS
    dom = day_of_month(close.index)
    rdt = close.index[dom.isin(DEFAULTS["tranche_days"]).to_numpy()]

    class Final:
        def __init__(self):
            self.b = TrendBook(close, {"symbols": list(close.columns)})
            for d in rdt[rdt < start]:
                self.b.rebalance(d)
            self.flips = 0

        def __call__(self, d):
            self.b.rebalance(d)
            return self.b.combined().drop(DEFAULTS["cash_symbol"], errors="ignore")
    fin = Final()
    ef, tof = simulate(close, rf, fin, rdt, open_, start=start)
    curves[FINAL] = ef
    res[FINAL] = {**metrics(ef, rf), "Switches/yr": float("nan"), "Turnover/yr": tof}
    eff, _ = simulate(close, rf, Final(), rdt, open_, cost=0.0025, start=start)
    res["Final design, costs 0.25%/side"] = {**metrics(eff, rf), "Switches/yr": float("nan"), "Turnover/yr": tof}
    e3, to3, fl3 = strat(signal="mom", lookback=12, band=0.0)
    res["12-mo momentum vs T-bills"] = {**metrics(e3, rf), "Switches/yr": fl3, "Turnover/yr": to3}
    spy, _ = simulate(close[["SPY"]], rf, Fixed({"SPY": 1.0}), rd, open_, start=start, once=True)
    curves["SPY buy & hold"] = spy
    res["SPY buy & hold"] = {**metrics(spy, rf), "Switches/yr": 0, "Turnover/yr": 0}
    sf, to6 = simulate(close[["SPY", "IEF"]], rf, Fixed({"SPY": .6, "IEF": .4}), rd, open_, start=start)
    curves["60/40 SPY/IEF"] = sf
    res["60/40 SPY/IEF"] = {**metrics(sf, rf), "Switches/yr": 0, "Turnover/yr": to6}

    class EW:
        flips = 0
        def __call__(self, d):
            a = [c for c in close.columns if pd.notna(close.at[d, c])]
            return pd.Series(1 / len(a), index=a)
    ew, to7 = simulate(close, rf, EW(), rd, open_, start=start)
    res["Same 9 assets, no trend rule"] = {**metrics(ew, rf), "Switches/yr": 0, "Turnover/yr": to7}
    ec, _, _ = strat(cost=0.0025)
    res["Trend, costs 0.25%/side (stress)"] = {**metrics(ec, rf), "Switches/yr": fl, "Turnover/yr": to}

    lines += [f"## {name}", f"{start.date()} → {close.index[-1].date()} "
              f"({len(ev(close)) / 252:.1f} years), fills: {'next-day open' if open_ is not None else 'next-day close'}, "
              f"costs {COST * 100:.2f}%/side, cash earns T-bills", "", fmt(pd.DataFrame(res).T), ""]

    # crash / bull windows
    rows = {}
    for lab, a, b in CRASHES:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if a < start or a > close.index[-1]:
            continue
        rows[lab] = {k: (v.loc[:b].iloc[-1] / v.loc[a:].iloc[0] - 1) * 100 for k, v in curves.items()}
    lines += ["### Crash and bull windows (total return %)", "", fmt(pd.DataFrame(rows).T, 1), ""]

    yr = pd.DataFrame({k: v.resample("YE").last() for k, v in curves.items()})
    yr = (yr.pct_change() * 100).dropna(how="all")
    yr.index = [f"{y} (YTD)" if y == close.index[-1].year else str(y) for y in yr.index.year]
    beat = (yr[FINAL] > yr["SPY buy & hold"]).mean() * 100
    lines += [f"### Calendar years — final design beat SPY in {beat:.0f}% of years", "", fmt(yr, 1), ""]

    if not deep:
        return curves, res, start

    # sensitivity grid
    grid = {}
    n_trials = 0
    for L, b in itertools.product([6, 8, 10, 12], [0.0, 0.005, 0.01, 0.02, 0.03]):
        eg, _, flg = strat(lookback=L, band=b)
        m = metrics(eg, rf)
        grid[(L, f"{b * 100:.1f}%")] = (m["Sharpe"], m["Max DD %"], m["CAGR %"], flg, eg)
        n_trials += 1
    n_trials += 2 * 2 * 4  # momentum + inv-vol variants and lookbacks, counted as tried
    sh = pd.DataFrame({k: v[0] for k, v in grid.items()}, index=["x"]).T["x"].unstack()
    dd = pd.DataFrame({k: v[1] for k, v in grid.items()}, index=["x"]).T["x"].unstack()
    lines += ["### Sensitivity: Sharpe by lookback (rows, months) and buffer (columns)", "", fmt(sh), "",
              "### Sensitivity: max drawdown %", "", fmt(dd, 1), ""]

    # timing luck
    tl = {}
    for k in list(range(0, 19)) + [-1]:
        rdk = rebalance_dates(close.index, k)
        sig = TrendSignal(close, rf, rdk, **BASE)
        ek, _ = simulate(close, rf, sig, rdk, open_, start=start)
        m = metrics(ek, rf)
        tl["last" if k == -1 else k + 1] = {"CAGR %": m["CAGR %"], "Sharpe": m["Sharpe"], "Max DD %": m["Max DD %"]}
    tl = pd.DataFrame(tl).T
    lines += ["### Rebalance-day timing luck (trend 10-mo/1%, rebalancing on each trading day of the month)", "",
              fmt(tl.describe().loc[["min", "25%", "50%", "75%", "max"]]), ""]

    # walk-forward
    years = sorted(set(ev(close).index.year))[3:]
    pieces, picks = [], {}
    for y in years:
        cut = pd.Timestamp(f"{y}-01-01")
        best = max(grid.items(), key=lambda kv: metrics(kv[1][4][kv[1][4].index < cut], rf)["Sharpe"])
        picks[y] = best[0]
        e_ = best[1][4]
        pieces.append(e_.pct_change()[(e_.index >= cut) & (e_.index < pd.Timestamp(f"{y + 1}-01-01"))])
    wf = (1 + pd.concat(pieces)).cumprod()
    fixed = curves["Trend (10-mo, 1% buffer)"]
    fixed = fixed[fixed.index >= wf.index[0]]
    spy_w = curves["SPY buy & hold"][lambda s: s.index >= wf.index[0]]
    wfr = {"Walk-forward (re-picks params yearly)": metrics(wf, rf), "Fixed 10-mo / 1%": metrics(fixed / fixed.iloc[0], rf),
           "SPY": metrics(spy_w / spy_w.iloc[0], rf)}
    lines += [f"### Walk-forward from {wf.index[0].date()} (params chosen only from earlier data)", "",
              fmt(pd.DataFrame(wfr).T), "", "Picks by year: " + ", ".join(f"{y}: {L}m/{b}" for y, (L, b) in picks.items()), ""]

    # statistics
    r = pd.DataFrame({k: v.pct_change() for k, v in curves.items()}).dropna()
    exr = r.sub(rf.reindex(r.index).fillna(0.0), axis=0)
    base_ex = exr[FINAL]
    psr = stats.psr(base_ex)
    dsr = stats.deflated_sharpe(base_ex, n_trials, 1.0 / (len(base_ex) - 1))
    boot = stationary_bootstrap(exr.to_numpy())
    cols = list(exr.columns)
    t, s, f = cols.index(FINAL), cols.index("SPY buy & hold"), cols.index("60/40 SPY/IEF")
    d_s, d_f = boot[:, t] - boot[:, s], boot[:, t] - boot[:, f]
    lines += ["### Statistics", "",
              "_Statistics are for the final design._",
              f"- PSR (probability true Sharpe > 0): **{psr:.2f}**",
              f"- Deflated Sharpe over {n_trials} variants tried: **{dsr:.2f}**",
              f"- Bootstrap Sharpe 90% range — trend {np.percentile(boot[:, t], 5):.2f} to {np.percentile(boot[:, t], 95):.2f}; "
              f"SPY {np.percentile(boot[:, s], 5):.2f} to {np.percentile(boot[:, s], 95):.2f}; "
              f"60/40 {np.percentile(boot[:, f], 5):.2f} to {np.percentile(boot[:, f], 95):.2f}",
              f"- P(trend Sharpe > SPY Sharpe) = **{(d_s > 0).mean():.0%}**; "
              f"P(trend Sharpe > 60/40 Sharpe) = **{(d_f > 0).mean():.0%}** (2,000 block-bootstrap resamples)", ""]
    return curves, res, start


def chart(curves, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    col = {FINAL: "#2a78d6", "SPY buy & hold": "#eb6834", "60/40 SPY/IEF": "#1baf7a"}
    lab = {FINAL: "Trend (final)", "SPY buy & hold": "SPY", "60/40 SPY/IEF": "60/40"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c9c8c2")
        ax.tick_params(colors="#52514e")
        ax.grid(axis="y", color="#ebeae5", lw=0.8)
    for k, c in col.items():
        e = curves[k] / curves[k].iloc[0] * 100000
        a1.plot(e.index, e, color=c, lw=2, label=lab[k])
        a1.annotate(f"{lab[k]}  ${e.iloc[-1] / 1000:,.0f}k", (e.index[-1], e.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color="#0b0b0b", fontsize=9)
        dd = (e / e.cummax() - 1) * 100
        a2.plot(dd.index, dd, color=c, lw=1.6)
    a1.set_yscale("log")
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
    a1.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    a1.set_title(title, loc="left", color="#0b0b0b", fontsize=12, pad=10)
    a1.legend(frameon=False, loc="upper left", labelcolor="#0b0b0b")
    a2.set_ylabel("Drawdown %", color="#52514e")
    a2.axhline(0, color="#c9c8c2", lw=0.8)
    fig.tight_layout()
    fig.subplots_adjust(right=0.86)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    etf_close, etf_open, ext_close, rf = load()
    lines = ["# ETF trend design — detailed backtest", "",
             "Sleeves: " + ", ".join(f"{n} ({e}; before launch {p})" for n, e, p in SLEEVES), ""]
    c1, _, _ = run_era("1. ETF era (real ETFs, next-day open fills)", etf_close.dropna(how="any"), rf,
                       etf_open.reindex(etf_close.dropna(how="any").index), lines)
    c2, _, s2 = run_era("2. Extended era (index-fund proxies before ETF launch, next-day close fills)",
                        ext_close, rf, None, lines, core=[e for e in ETFS if e != "GLD"])
    chart(c1, os.path.join(OUT, "etf_era.png"), "ETF era: $100k, next-day open fills, 0.10%/side costs")
    chart(c2, os.path.join(OUT, "extended_era.png"), f"Extended era from {s2.year}: index-fund proxies before ETF launch")
    pd.DataFrame(c2).to_csv(os.path.join(OUT, "extended_equity.csv"))
    path = os.path.join(OUT, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
