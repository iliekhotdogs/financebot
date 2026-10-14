#!/usr/bin/env python3
"""
Experiment: does the Markov regime method (jackson-video-resources/
markov-hedge-fund-method, framework by @RohOnChain) add value to this bot?

Uses the bot's own machinery so results are comparable to the learning loop:
GenomeStrategy backtests (next-open fills, 0.15%/side costs, ATR stops),
equal-weight PortfolioRun, the configured train/validation/OOS windows, and
PSR / deflated Sharpe from learning/stats.py. Data: yfinance, auto-adjusted.

Markov model (identical to markov_regime.py): label each day Bear/Sideways/Bull
from the trailing 20-day return (±5%), MLE 3x3 transition matrix from labels
up to day t only (causal, expanding), signal = P(Bull|state) - P(Bear|state).

Variants, all scored on validation+OOS (the learning loop's "scored" window):
  active             sma_cross 20/50 v1.1 (the live strategy)
  +sym_markov        active, entries only when the symbol's Markov signal > 0
  +spy_markov        active, entries only when SPY's Markov signal > 0
  +spy_not_bear      active, entries blocked only while SPY is labelled Bear
  markov_long        standalone: long while symbol signal > 0 (same costs/stops)
  plain_momentum     long while trailing 20d return > +5% (no matrix at all)
  buy_and_hold       equal-weight
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BOT_DIR)

from strategies.families import GenomeStrategy, DEFAULT_GENOME  # noqa: E402
from learning.loop import PortfolioRun, buy_and_hold_daily, window_bounds, _naive_daily_index  # noqa: E402
from learning import stats  # noqa: E402
from strategies.v1_1 import metrics_from_daily  # noqa: E402

SYMBOLS = ["AVGO", "COST", "JNJ", "AAPL", "AMZN", "GOOGL", "HD", "JPM", "LLY", "ABBV",
           "META", "MSFT", "PG", "TSLA", "XOM", "MA", "MRK", "NVDA", "UNH", "V"]  # learning run 2026-10-05
WINDOW, THRESH = 20, 0.05


# ---------------- Markov (ported 1:1 from markov_regime.py, made causal per day) -------------
def labels(close: pd.Series) -> pd.Series:
    r = close.pct_change(WINDOW)
    lab = pd.Series(1, index=close.index, dtype=int)
    lab[r > THRESH] = 2
    lab[r < -THRESH] = 0
    return lab.where(r.notna())


def causal_signal(close: pd.Series, min_train: int = 252) -> pd.Series:
    """signal[t] = P_t[state_t, Bull] - P_t[state_t, Bear], P_t from labels[:t+1] only."""
    lab = labels(close)
    arr = lab.to_numpy()
    out = np.full(len(arr), np.nan)
    counts = np.zeros((3, 3))
    seen = 0
    for t in range(1, len(arr)):
        a, b = arr[t - 1], arr[t]
        if a == a and b == b:
            counts[int(a), int(b)] += 1
            seen += 1
        if b == b and seen >= min_train:
            row = counts[int(b)]
            s = row.sum()
            if s > 0:
                out[t] = (row[2] - row[0]) / s
    return pd.Series(out, index=close.index)


class Filtered(GenomeStrategy):
    """Active strategy with an extra entry mask (exits unchanged)."""

    def __init__(self, genome, mask_fn):
        super().__init__(genome)
        self.mask_fn = mask_fn

    def _signals(self, df):
        entry, exit_ = super()._signals(df)
        m = self.mask_fn(df).reindex(df.index).fillna(False).to_numpy(bool)
        return entry & m, exit_


class StateLong(GenomeStrategy):
    """Long while cond is true: enter on cond turning on, exit when it turns off."""

    def __init__(self, cond_fn):
        super().__init__({**DEFAULT_GENOME, "version": "state_long"})
        self.cond_fn = cond_fn

    @property
    def min_bars(self):
        return 60

    def _signals(self, df):
        c = self.cond_fn(df).reindex(df.index).fillna(False)
        entry = c & ~c.shift(1, fill_value=False)
        return entry.to_numpy(bool), (~c).to_numpy(bool)


def load(symbols, days=1500):
    import yfinance as yf
    raw = yf.download(symbols, period="10y", auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)
    out = {}
    for s in symbols:
        try:
            df = raw[s][["Open", "High", "Low", "Close", "Volume"]].dropna()
        except KeyError:
            continue
        if len(df) > 300:
            out[s] = df
    return out


def main():
    cfg = yaml.safe_load(open(os.path.join(BOT_DIR, "config.yaml")))
    allbars = load(SYMBOLS + ["SPY"])
    spy = allbars.pop("SPY")["Close"]
    bars = allbars
    starts = window_bounds(bars, cfg)
    spy_sig = causal_signal(spy)
    spy_lab = labels(spy)

    def align(s, df):
        s = _naive_daily_index(s)
        idx = _naive_daily_index(pd.Series(0, index=df.index)).index
        return pd.Series(s.reindex(idx).to_numpy(), index=df.index)

    variants = {
        "active (v1.1 sma 20/50)": GenomeStrategy(DEFAULT_GENOME),
        "+ symbol Markov filter": Filtered(DEFAULT_GENOME, lambda df: causal_signal(df["Close"]) > 0),
        "+ SPY Markov filter": Filtered(DEFAULT_GENOME, lambda df: align(spy_sig, df) > 0),
        "+ SPY not-Bear filter": Filtered(DEFAULT_GENOME, lambda df: align(spy_lab, df) != 0),
        "Markov standalone (long-only)": StateLong(lambda df: causal_signal(df["Close"]) > 0),
        "plain 20d momentum >5% (no matrix)": StateLong(lambda df: df["Close"].pct_change(WINDOW) > THRESH),
    }

    bh = buy_and_hold_daily(bars)
    for label, scored_start in (("SCORED WINDOW (validation+OOS, as the learning loop)", starts["validation"]),
                                ("ROBUSTNESS: full history after 1y Markov warm-up",
                                 bh.index[252 + WINDOW])):
        print(f"\n=== {label} ===")
        report(variants, bars, bh, scored_start)

    agreement(bars, spy, spy_sig)


def report(variants, bars, bh, scored_start):
    rows, series = [], {}
    for name, strat in variants.items():
        run = PortfolioRun(strat, bars)
        d = run.daily[run.daily.index >= scored_start]
        w = run.window(scored_start)
        series[name] = d
        rows.append((name, d, w["trades"]))
    b = bh[bh.index >= scored_start]
    series["buy & hold (equal weight)"] = b
    rows.append(("buy & hold (equal weight)", b, 0))

    n_trials = 10 + len(variants)  # learning loop had tried 10; plus these
    var = stats.annual_to_daily_var([metrics_from_daily(d)["sharpe_ratio"] for _, d, _ in rows])
    print(f"Scored window: {scored_start.date()} → {b.index[-1].date()}  ({len(b)} days), "
          f"{len(bars)} symbols, costs 0.15%/side, next-open fills")
    print(f"{'variant':38s} {'Sharpe':>7s} {'Return':>8s} {'MaxDD':>7s} {'Trades':>7s} {'PSR':>5s} {'DSR':>5s}")
    for name, d, tr in rows:
        m = metrics_from_daily(d)
        print(f"{name:38s} {m['sharpe_ratio']:7.2f} {m['total_return_pct']:7.1f}% {abs(m['max_drawdown_pct']):6.1f}% "
              f"{tr:7d} {stats.psr(d):5.2f} {stats.deflated_sharpe(d, n_trials, var):5.2f}")
    report.last = series


def agreement(bars, spy, spy_sig):
    # How much does the matrix add over just reading the current label?
    agree, total = 0, 0
    for s, df in bars.items():
        sig = causal_signal(df["Close"]).dropna()
        lab = labels(df["Close"]).reindex(sig.index)
        naive = lab.map({2: 1, 0: -1, 1: 0})
        nz = naive != 0
        agree += int((np.sign(sig[nz]) == naive[nz]).sum())
        total += int(nz.sum())
    print(f"\nWhen the label is Bull/Bear, sign(Markov signal) == 'current label persists' "
          f"on {agree / total * 100:.1f}% of days.")
    d = labels(spy).dropna().astype(int)
    P = np.zeros((3, 3))
    for a, c in zip(d.iloc[:-1], d.iloc[1:]):
        P[a, c] += 1
    P = P / P.sum(1, keepdims=True)
    print("SPY transition matrix (rows from, cols to: Bear Side Bull):")
    for i, n in enumerate(["Bear", "Side", "Bull"]):
        print(f"  {n:5s} " + "  ".join(f"{x * 100:5.1f}%" for x in P[i]))
    print(f"SPY now: {['Bear', 'Sideways', 'Bull'][int(d.iloc[-1])]}, signal {spy_sig.dropna().iloc[-1]:+.3f}")

    out = pd.DataFrame({k: (1 + v).cumprod() for k, v in report.last.items()})
    out.to_csv(os.path.join(BOT_DIR, "reports", "markov_experiment_equity.csv"))


if __name__ == "__main__":
    main()
