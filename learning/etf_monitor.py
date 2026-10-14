"""
Daily monitoring for the ETF trend strategy (replaces the strategy search in
etf_trend mode — walk-forward tests showed re-tuning the parameters made
results worse, so the learning stage watches instead of re-optimising).

Answers three questions every day, on the ACTUAL paper account:
  1. Is the bot doing what the strategy says?  Tracking error between the
     paper equity and the model (strategies.etf_trend.model_equity) over the
     same days. Large tracking error = an execution or data problem.
  2. Is it delivering its promise?  Drawdown and Sharpe vs SPY and 60/40.
  3. Real-money readiness (learning/readiness.assess_etf).
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from strategies import etf_trend as et
from strategies.families import metrics_from_daily

BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQUITY_CURVE_PATH = os.path.join(BOT_DIR, "journal", "equity_curve.csv")
REPORT_DIR = os.path.join(BOT_DIR, "learning", "reports")


def config_label(config: Optional[dict] = None) -> str:
    """Strategy label the bot runs under (et.VERSION, plus the overlay suffix
    when etf_trend.overlay is enabled). Reads config.yaml when not given."""
    from strategies import overlay as ov
    if config is None:
        import yaml
        with open(os.path.join(BOT_DIR, "config.yaml")) as f:
            config = yaml.safe_load(f) or {}
    import ai_pm
    return ai_pm.label(config, ov.label(config.get("etf_trend") or {}, et.VERSION))


def paper_equity(path: Optional[str] = None, version: Optional[str] = None) -> pd.Series:
    """Daily paper equity for the ETF strategy only (rows written by etf cycles).
    version defaults to the label in config.yaml (see config_label)."""
    version = version or config_label()
    path = path or EQUITY_CURVE_PATH
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r.get("strategy_version") == version and r.get("equity")]
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series([float(r["equity"]) for r in rows], index=pd.to_datetime([r["date"] for r in rows]))
    return s.sort_index()


def compare(paper: pd.Series, closes: pd.DataFrame, opens: Optional[pd.DataFrame], params: dict) -> dict:
    """Paper vs model vs SPY vs 60/40 over the paper period."""
    out = {"days": max(0, len(paper) - 1)}
    if len(paper) < 3:
        return out
    c = closes.copy()
    c.index = pd.DatetimeIndex(c.index).normalize()
    start = paper.index[0]
    if start < c.index[0] or start > c.index[-1]:
        return out
    model = et.model_equity(c, opens, start, params)
    days = paper.index.intersection(model.index)
    if len(days) < 3:
        return out
    pr = paper.reindex(days).pct_change().dropna()
    mr = model.reindex(days).pct_change().dropna()
    spy = c["SPY"].reindex(days).pct_change().dropna() if "SPY" in c else None
    sf = None
    if "SPY" in c and "IEF" in c:
        r = c[["SPY", "IEF"]].reindex(days).pct_change().dropna()
        sf = r["SPY"] * 0.6 + r["IEF"] * 0.4
    m = lambda r: metrics_from_daily(r) if r is not None and len(r) > 1 else {}
    out.update(
        tracking_error_pct=float((pr - mr).std() * np.sqrt(252) * 100) if len(pr) > 2 else None,
        paper=m(pr), model=m(mr), spy=m(spy), sixty_forty=m(sf),
    )
    return out


def report(dp, config: dict, now: Optional[datetime] = None) -> tuple:
    """Writes learning/reports/<run>.md; returns (path, comparison, readiness)."""
    from engine_etf import load_prices
    from learning.etf_fills import reconcile_fills

    now = now or datetime.now(timezone.utc)
    params = et.params_from_config(config)
    syms = list(dict.fromkeys(list(params["symbols"]) + [params["cash_symbol"], "SPY", "IEF"]))
    closes, opens = load_prices(dp, syms, now)
    targets, info = et.targets_asof(closes, None, params)
    label = config_label(config)
    import ai_pm
    ai_on = ai_pm.active(config)
    paper = paper_equity()
    cmp_ = compare(paper, closes, opens, params)
    from learning.readiness import assess_etf_and_notify
    rd = assess_etf_and_notify(config, paper, cmp_)

    def row(name, mm):
        if not mm:
            return f"| {name} | — | — | — |"
        return (f"| {name} | {mm.get('total_return_pct', 0):.2f}% | {mm.get('sharpe_ratio', 0):.2f} | "
                f"{abs(mm.get('max_drawdown_pct', 0)):.1f}% |")

    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    lines = [f"# ETF trend monitor {run_id}", "",
             f"Strategy {label}: {len(params['symbols'])} ETFs, {params['lookback_months']}-month average, "
             f"±{params['band'] * 100:.0f}% buffer, {params['weighting']} weights, "
             f"{len(params['tranche_days'])} tranches. Data through {info['asof']}.", "",
             "## Targets", "", "| ETF | Target | Tranches in market |", "|---|---|---|"]
    for s, w in targets.sort_values(ascending=False).items():
        lines.append(f"| {s} | {w * 100:.1f}% | {info['in_market'].get(s, '—')}/{info['tranches']} |")
    lines += ["", f"## Paper vs model ({cmp_.get('days', 0)} days)", ""]
    if cmp_.get("tracking_error_pct") is not None:
        te = cmp_["tracking_error_pct"]
        if ai_on:
            diff = cmp_["paper"].get("total_return_pct", 0) - cmp_["model"].get("total_return_pct", 0)
            head = (f"- AI vs rules: **{diff:+.2f}%** total return (paper with AI minus the rules alone); "
                    f"difference in daily returns {te:.2f}% a year")
        else:
            flag = "OK" if te <= float(config.get("live_readiness", {}).get("max_tracking_error_pct", 3.0)) else "INVESTIGATE"
            head = f"- Tracking error vs model: **{te:.2f}%** a year ({flag})"
        lines += [head, "",
                  "| | Return | Sharpe | Max drawdown |", "|---|---|---|---|",
                  row("Paper account", cmp_["paper"]), row("Rules only, no AI (model)" if ai_on else "Model (what it should have done)", cmp_["model"]),
                  row("SPY", cmp_["spy"]), row("60/40 SPY/IEF", cmp_["sixty_forty"]), ""]
    else:
        lines += ["- Not enough paper days on this strategy yet.", ""]
    lines += ["## Real-money readiness", f"- **{rd['status']}** — {rd['summary']}", ""]
    fills = reconcile_fills(dp)
    slippage = [float(r["slippage_bp"]) for r in fills]
    if slippage:
        lines.append(f"- ETF fills: {len(slippage)} fills, average {sum(slippage) / len(slippage):.2f} bp, "
                     f"worst {max(slippage):.2f} bp")
    else:
        lines.append("- ETF fills: 0 fills, average n/a, worst n/a")
    lines.append("")
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"{run_id}.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    # Display-only snapshot; keep compare() and the monitoring decisions unchanged.
    import json
    series = {key: {} for key in ("paper", "model", "spy", "sixty_forty")}
    if len(paper) >= 3 and cmp_.get("model"):
        model = et.model_equity(closes, opens, paper.index[0], params)
        days = paper.index.intersection(model.index)
        if len(days) >= 3:
            c = closes.copy()
            c.index = pd.DatetimeIndex(c.index).normalize()
            curves = {"paper": paper.reindex(days), "model": model.reindex(days)}
            if "SPY" in c:
                curves["spy"] = c["SPY"].reindex(days)
            if "SPY" in c and "IEF" in c:
                returns = c[["SPY", "IEF"]].reindex(days).pct_change().fillna(0)
                curves["sixty_forty"] = (1 + returns["SPY"] * .6 + returns["IEF"] * .4).cumprod()
            for key, values in curves.items():
                values = values.dropna()
                if len(values) and values.iloc[0]:
                    series[key] = {str(d.date()): round(float(v / values.iloc[0] * 100), 6)
                                   for d, v in values.items()}
    payload = {"generated_at": now.isoformat(), "tracking_error_pct": cmp_.get("tracking_error_pct"),
               **{k: cmp_.get(k, {}) for k in ("paper", "model", "spy", "sixty_forty")},
               "series": series}
    compare_path = os.path.join(BOT_DIR, "learning", "etf_compare.json")
    os.makedirs(os.path.dirname(compare_path), exist_ok=True)
    tmp = compare_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, allow_nan=False)
    os.replace(tmp, compare_path)
    return path, cmp_, rd
