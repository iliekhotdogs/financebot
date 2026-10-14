#!/usr/bin/env python3
"""
Does the learning loop actually learn? Replay it through history.

The REAL learning code (learning/loop.py: propose → evaluate → deflated-
Sharpe gates → shadow paper → announce → auto-switch) is run every
`--step` trading days on data truncated at the simulated date — it never
sees the future. Between runs the portfolio trades whatever strategy was
active, and the stitched equity curve is compared with:
  - static : never learning (keep the starting strategy)
  - B&H    : equal-weight buy-and-hold of the same symbols

Modes
  real   yfinance daily history (default 12 liquid symbols, 10y)
  trend  synthetic: persistent up/down regimes with ZERO average drift —
         a real, learnable trend-following edge exists; the incumbent is a
         mean-reversion strategy that is wrong for this market. A working
         learner must switch category and beat static.
  null   synthetic: i.i.d. returns, no edge anywhere. A working learner
         must NOT promote noise (≈ zero switches, no fake outperformance).

Usage
  python scripts/simulate_learning.py --mode trend
  python scripts/simulate_learning.py --mode null
  python scripts/simulate_learning.py --mode real --years 10
"""
import argparse
import json
import os
import sys
import tempfile
from datetime import timezone

import numpy as np
import pandas as pd
import yaml

BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BOT_DIR)
os.environ["FINBOT_NOTIFY"] = "off"

import learning.loop as loop  # noqa: E402
import notify  # noqa: E402
from strategies import families as fam  # noqa: E402
from learning import stats  # noqa: E402


def synthetic(mode: str, n_symbols: int, n_days: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-02", periods=n_days, tz="UTC")
    bars = {}
    for k in range(n_symbols):
        if mode == "trend":
            drift = np.empty(n_days)
            i = 0
            sign = rng.choice([-1, 1])
            while i < n_days:
                L = int(rng.integers(40, 120))
                drift[i:i + L] = sign * rng.uniform(0.0015, 0.003)
                sign = -sign
                i += L
            r = drift + rng.normal(0, 0.012, n_days)
        else:
            r = rng.normal(0.0, 0.015, n_days)
        c = 100 * np.exp(np.cumsum(r))
        o = c * np.exp(rng.normal(0, 0.003, n_days))
        hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n_days)))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n_days)))
        bars[f"S{k:02d}"] = pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": c,
                                          "Volume": 1e6}, index=idx)
    return bars


def real(symbols: list, years: int) -> dict:
    import yfinance as yf
    out = {}
    for s in symbols:
        df = yf.Ticker(s).history(period=f"{years}y", interval="1d", auto_adjust=True)
        if len(df) > 600:
            out[s] = df[["Open", "High", "Low", "Close", "Volume"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "trend", "null"], default="trend")
    ap.add_argument("--symbols", type=int, default=12)
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--step", type=int, default=21, help="trading days between learning runs")
    ap.add_argument("--propose", type=int, default=6, help="candidates proposed per run")
    ap.add_argument("--explore", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--incumbent", default=None, help="starting family (default: config / mean_reversion for trend)")
    ap.add_argument("--json", default=None, help="write results JSON here")
    args = ap.parse_args()

    with open(os.path.join(BOT_DIR, "config.yaml")) as f:
        config = yaml.safe_load(f)
    lc = config.setdefault("learning", {})
    lc["propose_per_run"] = args.propose
    lc["explore_per_run"] = args.explore
    lc.setdefault("promotion", {}).update({"allow_auto_promotion": True, "announce_grace_hours": 0})

    if args.mode == "real":
        syms = config["market_universe"]["symbols"][:args.symbols]
        bars = real(syms, args.years)
    else:
        bars = synthetic(args.mode, args.symbols, 252 * args.years, args.seed)
    if not bars:
        sys.exit("no data")

    incumbent = args.incumbent or ("mean_reversion" if args.mode == "trend" else None)
    if incumbent:
        config["strategy"] = {"family": incumbent}
    start_genome = loop.active_genome(config) if not incumbent else fam.normalize({"family": incumbent})

    tmp = tempfile.mkdtemp(prefix="simlearn_")
    for name in ("CANDIDATES_PATH", "AUDIT_PATH", "EVALUATIONS_PATH", "ACTIVE_STRATEGY_PATH",
                 "PENDING_PROMOTION_PATH"):
        setattr(loop, name, os.path.join(tmp, name.lower()))
    loop.REPORTS_DIR = os.path.join(tmp, "reports")
    announcements = []
    notify.send = lambda cfg, text, kind="info": announcements.append(text) or True

    cal = sorted(set().union(*[set(df.index) for df in bars.values()]))
    ev = config["learning"]["evaluation"]
    hist = int(ev.get("history_days", 1500))
    first = int(hist / 1.45)             # ≈ trading days in the learning history window
    run_dates = cal[first::args.step]
    timeline = [(cal[first], start_genome)]
    print(f"[{args.mode}] {len(bars)} symbols, {len(cal)} days, {len(run_dates)} learning runs, "
          f"incumbent {fam.describe(start_genome)}")

    for k, t in enumerate(run_dates):
        window = {s: df[(df.index <= t) & (df.index > t - pd.Timedelta(days=hist))] for s, df in bars.items()}
        window = {s: df for s, df in window.items() if len(df) >= 200}
        now = pd.Timestamp(t).tz_convert("UTC").to_pydatetime().replace(hour=21) if pd.Timestamp(t).tzinfo \
            else pd.Timestamp(t).to_pydatetime().replace(hour=21, tzinfo=timezone.utc)
        loop.run_learning_cycle(None, config, list(window), now=now, bars=window, seed=args.seed * 1000 + k)
        g = loop.active_genome(config)
        if fam.genome_hash(g) != fam.genome_hash(timeline[-1][1]):
            timeline.append((t, g))
            print(f"  {pd.Timestamp(t).date()}  SWITCH → {fam.describe(g)}")

    # --- stitch realised returns: each segment trades the genome active then ---
    def naive(s):
        s = s.copy()
        idx = pd.to_datetime(s.index)
        s.index = (idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx).normalize()
        return s

    t0 = naive(pd.Series(index=[timeline[0][0]], dtype=float)).index[0]
    learned = []
    for i, (ts, g) in enumerate(timeline):
        a = naive(pd.Series(index=[ts], dtype=float)).index[0]
        b = naive(pd.Series(index=[timeline[i + 1][0]], dtype=float)).index[0] if i + 1 < len(timeline) else None
        d = loop.PortfolioRun(fam.make_strategy(g), bars).daily
        d = d[d.index > a]
        if b is not None:
            d = d[d.index <= b]
        learned.append(d)
    learned = pd.concat(learned)
    static = loop.PortfolioRun(fam.make_strategy(start_genome), bars).daily
    static = static[static.index > t0]
    bh = loop.buy_and_hold_daily(bars)
    bh = bh[bh.index > t0]

    def summary(d):
        m = fam.metrics_from_daily(d)
        yrs = max(len(d) / 252, 1e-9)
        cagr = ((1 + m["total_return_pct"] / 100) ** (1 / yrs) - 1) * 100
        return {"CAGR%": round(cagr, 2), "Sharpe": m["sharpe_ratio"], "MaxDD%": abs(m["max_drawdown_pct"]),
                "PSR": round(stats.psr(d), 3)}

    tried = len(loop.CandidateStore(loop.CANDIDATES_PATH).read_all())
    res = {"mode": args.mode, "symbols": len(bars), "runs": len(run_dates), "strategies_tried": tried,
           "switches": len(timeline) - 1,
           "timeline": [(str(pd.Timestamp(t).date()), fam.describe(g)) for t, g in timeline],
           "learned": summary(learned), "static": summary(static), "buy_and_hold": summary(bh),
           "sim_dir": tmp}
    print(f"\n=== Result ({args.mode}) — {tried} strategies tried, {res['switches']} switches ===")
    for k in ("learned", "static", "buy_and_hold"):
        print(f"  {k:13s} {res[k]}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2, default=str)
    return res


if __name__ == "__main__":
    main()
