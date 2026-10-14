#!/usr/bin/env python3
"""
AI dry run: ask the AI (etf_trend.ai, Kimi K3 by default) once for a plan on the
latest data and print the briefing, the raw answer and the resulting targets.
Saves nothing, posts nothing to Discord, places no orders.

Usage: .venv/bin/python scripts/ai_decide.py
"""
import json
import os
import sys

import yaml

BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BOT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BOT_DIR, ".env"), override=False)
load_dotenv(os.getenv("HERMES_ENV_FILE", "/opt/data/.env"), override=False)


def main():
    """Ask the AI once for a plan on the latest data and print it. Saves nothing,
    posts nothing to Discord, places no orders (a test of the real model call)."""
    import ai_pm
    from data import DataProvider
    from strategies import etf_trend as et
    from engine_etf import load_prices, decide, max_exposure, _peak_equity
    with open(os.path.join(BOT_DIR, "config.yaml")) as f:
        config = yaml.safe_load(f)
    dp = DataProvider(paper=os.getenv("APCA_PAPER", "true").lower() == "true")
    params = et.params_from_config(config)
    cfg = config.get("etf_trend") or {}
    closes, _ = load_prices(dp, list(params["symbols"]) + [params["cash_symbol"]])
    account, positions = {}, []
    if getattr(dp, "trading_client", None):
        account, positions = dp.get_account(), dp.get_positions()
    equity = float(account.get("equity") or 0) or 100_000.0
    cash = float(account.get("cash") or 0)
    holdings = {p["symbol"]: float(p.get("qty") or 0) for p in positions}
    prices = {s: float(closes[s].dropna().iloc[-1]) for s in closes.columns}
    d = decide(closes, holdings, prices, equity, cash, {"initialized": True}, cfg, params)
    current_w = {s: q * prices.get(s, 0) / equity for s, q in holdings.items()}
    r = ai_pm.dry_run(config, asof=d["asof"], rules=d["rules_targets"], info=d["info"], closes=closes,
                      current_w=current_w, equity=equity, cash=cash, peak=_peak_equity(equity),
                      cap=max_exposure(cfg), symbols=list(params["symbols"]))
    s = ai_pm.settings(config)
    print(f"AI dry run — {s['model']}, data through {d['asof']}, briefing {r['briefing_chars']} chars "
          f"(~{r['briefing_chars'] // 4} tokens)")
    print("Briefing: " + json.dumps(r["briefing"], separators=(",", ":")))
    print(f"Raw answer: {json.dumps(r['raw'])}")
    if r["error"]:
        print(f"RESULT: no usable answer ({r['error']}) — the live bot would keep the current plan.")
        return
    print(f"RESULT: valid. {ai_pm._fmt(r['plan'])} · confidence {r['plan']['confidence']:.0%}")
    print(f"Summary: {r['plan']['summary']}")
    print("Rules:   " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in d["rules_targets"].sort_values(ascending=False).items()))
    print("With AI: " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in r["final"].sort_values(ascending=False).items()))
    print("(dry run: nothing saved, nothing sent, no orders)")


if __name__ == "__main__":
    main()
