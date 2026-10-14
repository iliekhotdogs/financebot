"""
Decision-policy learning — does the LLM layer (Jev + advisor) add value?

Every entry signal is recorded as a forecast tagged [taken] / [vetoed] /
[risk_rejected] / [skipped] with Jev's confidence. Once the horizon has
passed, the realised forward return is measured from price data and the
policy is re-fit:

  min_confidence_entry  threshold maximising mean realised return of the
                        signals it would have let through (0 = take all)
  jev_can_veto          False when signals Jev vetoed did significantly
                        BETTER than the ones it took (Jev is anti-predictive);
                        the engine then follows the strategy signal and only
                        uses Jev for logging.

Changes require a minimum sample and a t-stat > 2 versus the "take every
signal" baseline, so noise doesn't swing the policy. Written to
learning/decision_policy.json (read by engine.py each cycle).
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .forecasts import ForecastStore

LEARNING_DIR = os.path.dirname(os.path.abspath(__file__))
FORECASTS_PATH = os.path.join(LEARNING_DIR, "forecasts.csv")
POLICY_PATH = os.path.join(LEARNING_DIR, "decision_policy.json")
TAGS = ("taken", "vetoed", "risk_rejected", "skipped")


def _tag(thesis: str) -> Optional[str]:
    for t in TAGS:
        if thesis.startswith(f"[{t}]"):
            return t
    return None


def realised_returns(forecasts: list, bars: dict, now: datetime) -> pd.DataFrame:
    """One row per matured, tagged forecast with its realised horizon return (%)."""
    rows = []
    for f in forecasts:
        tag = _tag(f.thesis or "")
        df = bars.get(f.symbol)
        if tag is None or df is None or df.empty:
            continue
        t0 = pd.Timestamp(datetime.fromisoformat(f.decision_time)).tz_convert("UTC").tz_localize(None)
        idx = pd.to_datetime(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        closes = pd.Series(df["Close"].to_numpy(float), index=idx)
        before = closes[closes.index <= t0 + pd.Timedelta(hours=23)]
        if before.empty:
            continue
        i0 = closes.index.get_loc(before.index[-1])
        i1 = i0 + int(f.horizon_days)
        if i1 >= len(closes):
            continue  # not matured yet
        rows.append({"symbol": f.symbol, "tag": tag, "confidence": float(f.confidence),
                     "ret": (closes.iloc[i1] / closes.iloc[i0] - 1) * 100,
                     "decision_time": f.decision_time})
    return pd.DataFrame(rows)


def _tstat(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else 0.0


def fit_policy(df: pd.DataFrame, min_samples: int = 30, current: Optional[dict] = None) -> dict:
    current = current or {}
    if df.empty or len(df) < min_samples:
        return {**current, "status": "insufficient_data", "samples": int(len(df))}
    sig = df[df["tag"].isin(["taken", "vetoed"])]
    base = sig["ret"].to_numpy()
    best_t, best_mean, best_ts = 0.0, float(np.mean(base)) if len(base) else 0.0, 0.0
    grid = {}
    for t in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        sub = sig[sig["confidence"] >= t]["ret"].to_numpy()
        if len(sub) < max(10, min_samples // 3):
            continue
        grid[t] = {"n": int(len(sub)), "mean": float(sub.mean())}
        ts = _tstat(sub, base)
        if sub.mean() > best_mean and ts > 2.0:
            best_t, best_mean, best_ts = t, float(sub.mean()), ts

    taken = sig[sig["tag"] == "taken"]["ret"].to_numpy()
    vetoed = sig[sig["tag"] == "vetoed"]["ret"].to_numpy()
    veto_t = _tstat(vetoed, taken)
    jev_can_veto = not (len(taken) >= min_samples // 2 and len(vetoed) >= min_samples // 2 and veto_t > 2.0)

    return {
        "status": "fitted",
        "min_confidence_entry": best_t if best_t > 0 else current.get("min_confidence_entry", 0.6),
        "jev_can_veto": jev_can_veto,
        "samples": int(len(sig)),
        "evidence": {
            "take_all_mean_ret_pct": round(float(np.mean(base)), 3) if len(base) else None,
            "taken_mean_ret_pct": round(float(taken.mean()), 3) if len(taken) else None,
            "vetoed_mean_ret_pct": round(float(vetoed.mean()), 3) if len(vetoed) else None,
            "vetoed_vs_taken_tstat": round(veto_t, 2),
            "best_threshold_tstat": round(best_ts, 2),
            "threshold_grid": grid,
        },
    }


def update_policy(config: dict, bars: dict, now: Optional[datetime] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    fc = config.get("learning", {}).get("decision_policy", {})
    current = {}
    if os.path.exists(POLICY_PATH):
        with open(POLICY_PATH) as f:
            current = json.load(f)
    df = realised_returns(ForecastStore(FORECASTS_PATH).read_all(), bars, now)
    policy = fit_policy(df, int(fc.get("min_samples", 30)), current)
    policy["updated_at"] = now.isoformat()
    if policy["status"] == "fitted" and fc.get("enabled", True):
        with open(POLICY_PATH, "w") as f:
            json.dump(policy, f, indent=2)
    return policy
