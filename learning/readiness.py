"""
Real-money readiness — judged on the ACTUAL paper account, never backtests.

Inputs: journal/equity_curve.csv (one equity value per day, written by every
cycle), journal/closed_trades.csv, and a benchmark (SPY) over the same days.

Criteria (config `live_readiness`, all must pass):
  min_days               trading days of paper record
  min_closed_trades      closed round trips
  min_sharpe             annualised Sharpe of daily equity returns
  max_drawdown_pct       peak-to-trough on the equity curve
  min_psr                P[true Sharpe > 0] (Probabilistic Sharpe Ratio)
  must_beat_benchmark    Sharpe above SPY buy-and-hold over the same days
  min_days_since_switch  current strategy stable for N days (a record built
                         by a different strategy says little about this one)

READY only means "worth a human review for real money". The bot never
switches itself to a live account: order placement refuses non-paper
credentials in code.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from strategies.families import metrics_from_daily
from . import stats

BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQUITY_CURVE_PATH = os.path.join(BOT_DIR, "journal", "equity_curve.csv")
CLOSED_TRADES_PATH = os.path.join(BOT_DIR, "journal", "closed_trades.csv")
STATE_PATH = os.path.join(BOT_DIR, "learning", "readiness_state.json")

DEFAULTS = {"min_days": 126, "min_closed_trades": 40, "min_sharpe": 1.0, "max_drawdown_pct": 10.0,
            "min_psr": 0.95, "must_beat_benchmark": True, "min_days_since_switch": 63}


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def assess(config: dict, benchmark_close: Optional[pd.Series] = None,
           equity_rows: Optional[list] = None, closed: Optional[list] = None) -> dict:
    crit = {**DEFAULTS, **config.get("live_readiness", {})}
    rows = equity_rows if equity_rows is not None else _read_csv(EQUITY_CURVE_PATH)
    closed = closed if closed is not None else _read_csv(CLOSED_TRADES_PATH)
    checks = {}
    if len(rows) < 2:
        return {"status": "NOT_READY", "days": len(rows), "checks": {"min_days": False},
                "summary": "no paper track record yet"}

    eq = pd.Series([float(r["equity"]) for r in rows],
                   index=pd.to_datetime([r["date"] for r in rows])).sort_index()
    rets = eq.pct_change().dropna()
    m = metrics_from_daily(rets)
    days = len(rets)
    versions = [r.get("strategy_version", "") for r in rows]
    cur = versions[-1]
    since_switch = 0
    for v in reversed(versions):
        if v != cur:
            break
        since_switch += 1

    bench_sharpe = None
    if benchmark_close is not None and len(benchmark_close) > 2:
        b = benchmark_close.copy()
        b.index = pd.to_datetime(b.index)
        if b.index.tz is not None:
            b.index = b.index.tz_convert("UTC").tz_localize(None)
        b.index = b.index.normalize()
        b = b[(b.index >= eq.index[0]) & (b.index <= eq.index[-1])]
        bench_sharpe = metrics_from_daily(b.pct_change().dropna())["sharpe_ratio"]

    psr = stats.psr(rets)
    checks["min_days"] = days >= crit["min_days"]
    checks["min_closed_trades"] = len(closed) >= crit["min_closed_trades"]
    checks["min_sharpe"] = m["sharpe_ratio"] >= crit["min_sharpe"]
    checks["max_drawdown_pct"] = abs(m["max_drawdown_pct"]) <= crit["max_drawdown_pct"]
    checks["min_psr"] = psr >= crit["min_psr"]
    if crit["must_beat_benchmark"]:
        checks["beats_benchmark"] = bench_sharpe is not None and m["sharpe_ratio"] > bench_sharpe
    checks["min_days_since_switch"] = since_switch >= crit["min_days_since_switch"]
    ready = all(checks.values())
    return {
        "status": "READY_FOR_REVIEW" if ready else "NOT_READY",
        "days": days, "closed_trades": len(closed), "sharpe": m["sharpe_ratio"],
        "return_pct": m["total_return_pct"], "max_drawdown_pct": abs(m["max_drawdown_pct"]),
        "psr": round(psr, 3), "benchmark_sharpe": bench_sharpe,
        "days_since_strategy_switch": since_switch, "strategy_version": cur,
        "checks": checks,
        "summary": ("all real-money criteria met — operator review recommended" if ready else
                    "failing: " + ", ".join(k for k, v in checks.items() if not v)),
    }


def assess_and_notify(config: dict, benchmark_close: Optional[pd.Series] = None) -> dict:
    """Assess; post to Discord when the status changes to READY_FOR_REVIEW."""
    result = assess(config, benchmark_close)
    prev = {}
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            prev = json.load(f)
    if result["status"] == "READY_FOR_REVIEW" and prev.get("status") != "READY_FOR_REVIEW":
        import notify
        notify.send(config, (
            "🏁 Paper results passed the review checks. Please review before considering "
            "real-money trading. The bot will not switch by itself."), kind="alerts")
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump({**result, "updated_at": datetime.now(timezone.utc).isoformat()}, f, indent=2,
                  default=str)
    return result


ETF_DEFAULTS = {"min_days": 252, "max_drawdown_pct": 20.0, "max_tracking_error_pct": 3.0,
                "drawdown_vs_60_40": True, "min_days_since_switch": 252}


def assess_etf(config: dict, paper: pd.Series, comparison: dict) -> dict:
    """
    Readiness for the ETF trend strategy (strategy.mode = etf_trend).

    The 20-year backtest is the evidence that the STRATEGY works; paper trading
    has to show that the BOT implements it faithfully and that it behaves as
    designed. A Sharpe/PSR bar on one year would need ~2.7 years of data
    (minimum track record length) and mostly measures luck.

      min_days               trading days on this strategy (also = days since switch)
      max_tracking_error_pct paper vs model daily returns, annualised
      max_drawdown_pct       paper drawdown cap
      drawdown_vs_60_40      paper drawdown no deeper than 60/40's over the same days
      drawdown_vs_spy        (optional) paper drawdown no deeper than SPY's — for the overlay
    """
    lr = config.get("live_readiness", {}) or {}
    crit = {**ETF_DEFAULTS, **(lr.get("etf_trend") or {})}
    days = int(comparison.get("days", 0))
    te = comparison.get("tracking_error_pct")
    pm, sf = comparison.get("paper") or {}, comparison.get("sixty_forty") or {}
    dd = abs(pm.get("max_drawdown_pct", 0.0)) if pm else None
    sf_dd = abs(sf.get("max_drawdown_pct", 0.0)) if sf else None
    ai_on = bool(((config.get("etf_trend") or {}).get("ai") or {}).get("enabled"))
    checks = {"min_days": days >= crit["min_days"]}
    if ai_on:
        # The AI deviates from the model on purpose: instead of tracking it, it has to beat it.
        mm = comparison.get("model") or {}
        checks["ai_beats_rules"] = bool(pm and mm) and pm.get("total_return_pct", 0) >= mm.get("total_return_pct", 0)
    else:
        checks["tracking_error"] = te is not None and te <= crit["max_tracking_error_pct"]
    checks["max_drawdown_pct"] = dd is not None and dd <= crit["max_drawdown_pct"]
    if crit.get("drawdown_vs_60_40", True):
        checks["drawdown_vs_60_40"] = dd is not None and sf_dd is not None and dd <= sf_dd + 0.5
    spy_m = comparison.get("spy") or {}
    spy_dd = abs(spy_m.get("max_drawdown_pct", 0.0)) if spy_m else None
    if crit.get("drawdown_vs_spy", False):    # index/leverage overlay: compare with SPY instead
        checks["drawdown_vs_spy"] = dd is not None and spy_dd is not None and dd <= spy_dd + 0.5
    checks["min_days_since_switch"] = days >= crit["min_days_since_switch"]
    ready = all(checks.values())
    result = {
        "mode": "etf_trend",
        "status": "READY_FOR_REVIEW" if ready else "NOT_READY",
        "days": days, "tracking_error_pct": None if te is None else round(te, 2),
        "max_drawdown_pct": dd, "sixty_forty_max_drawdown_pct": sf_dd, "spy_max_drawdown_pct": spy_dd,
        "sharpe": pm.get("sharpe_ratio") if pm else None,
        "benchmark_sharpe": (comparison.get("spy") or {}).get("sharpe_ratio"),
        "sixty_forty_sharpe": sf.get("sharpe_ratio") if sf else None,
        "return_pct": pm.get("total_return_pct") if pm else None,
        "days_since_strategy_switch": days, "targets": crit, "checks": checks,
        "summary": ("all real-money criteria met — operator review recommended" if ready else
                    "failing: " + ", ".join(k for k, v in checks.items() if not v)),
    }
    return result


def assess_etf_and_notify(config: dict, paper: pd.Series, comparison: dict) -> dict:
    result = assess_etf(config, paper, comparison)
    prev = {}
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            prev = json.load(f)
    if result["status"] == "READY_FOR_REVIEW" and prev.get("status") != "READY_FOR_REVIEW":
        import notify
        notify.send(config, (
            "🏁 ETF paper results passed the review checks. Please review before considering "
            "real-money trading. The bot will not switch by itself."), kind="alerts")
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump({**result, "updated_at": datetime.now(timezone.utc).isoformat()}, f, indent=2, default=str)
    return result
