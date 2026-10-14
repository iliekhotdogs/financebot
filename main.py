#!/usr/bin/env python3
"""
Combined Trading Bot — Alpaca + yfinance (paper) / Liquid Trade (live phase 2).

Decision flow (trade command):
  1. Collect timestamped market/account/position/strategy snapshot
  2. Send snapshot to Advisor model → get advice (recommendation + evidence + risks)
  3. Send snapshot + valid actions + advisor output to Jev → final decision
  4. Pass Jev's decision through RiskManager (mandatory, can reject/resize)
  5. Only after risk approval → place paper order via Alpaca
  6. Log full decision chain to audit_log.csv

Paper trading mode:  yfinance (backfill) + Alpaca (real-time + paper execution).
Live trading mode:    Liquid Trade perps — enabled only after paper trading
                      is validated and operator approves.

Usage:
    python main.py [command]

Commands:
    health        - Check data connectivity (yfinance + Alpaca + Liquid)
    backfill      - Backfill historical data for symbols in config
    backtest      - Run a backtest on historical data
    run           - Run one monitoring cycle (signal + advisor + Jev + risk checks, NO order placement)
    trade         - Run full decision flow + place paper trades via Alpaca (after Jev + risk pass)
    cycle         - Scheduled entry point: `trade` if execution.auto_trade is true, else `run`
    learn         - Recursive learning loop: propose → walk-forward → shadow paper → select
    learning-approve  - Make a READY candidate the active strategy (operator approval)
    learning-rollback - Revert to the previously active strategy
    account       - Show Alpaca account state
    review        - Generate daily review report (includes advisor + Jev audit trail)
    liquid-status - Show Liquid Trade integration status (disabled by default)
    research      - Run research roles on current snapshot (technical/fundamental/bull-bear)
    learning-status    - Show learning system status (candidates, evaluations, promotions)
    learning-evaluate  - Run walk-forward evaluation on candidates
    learning-promote   - Assess promotion readiness for paper-trading candidates
    learning-select    - Run model selection across all candidates
    dashboard          - Start web dashboard on localhost:8080
"""

import os
import json
import yaml
import logging
import argparse
from datetime import datetime, timedelta, timezone
import pandas as pd

from dotenv import load_dotenv
# Process env wins; then the project .env; then the Hermes vault
# (/opt/data/.env, where keys actually live on Hermes cloud).
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=False)
load_dotenv(os.getenv("HERMES_ENV_FILE", "/opt/data/.env"), override=False)

from data import DataProvider
from risk import RiskManager
from liquid.client import LiquidClient
from journal.trade_log import ensure_header
from journal.audit_log import ensure_audit_header
from snapshot import build_snapshot
from engine import run_cycle, resolve_market_universe, strategy_from_config  # noqa: F401 (re-exported)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("finance-bot")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def cmd_health(args, dp: DataProvider, config):
    """Check connectivity to both data sources."""
    print("=== Health Check ===")

    # yfinance test
    print("\n[1] yfinance (historical/backfill)...")
    try:
        df = dp.fetch_yf_history("AAPL", period="5d", interval="1d")
        if not df.empty:
            print(f"  ✓ OK — AAPL returned {len(df)} rows, latest close: {df['Close'].iloc[-1]:.2f}")
        else:
            print("  ✗ FAIL — no data returned")
    except Exception as e:
        print(f"  ✗ FAIL — {e}")

    # Alpaca test
    print("\n[2] Alpaca (real-time + trading)...")
    if dp.trading_client:
        try:
            account = dp.get_account()
            print(f"  ✓ OK — Account status: {account.get('status', 'unknown')}")
            print(f"    Cash: {account.get('cash', 'N/A')}")
            clock = dp.get_clock()
            print(f"    Market open: {clock.get('is_open', 'unknown')}")
        except Exception as e:
            print(f"  ✗ FAIL — {e}")

        try:
            quote = dp.get_latest_quote("AAPL")
            print(f"  ✓ Quote OK — AAPL bid: {quote.get('bid_price', 'N/A')}, ask: {quote.get('ask_price', 'N/A')}")
        except Exception as e:
            print(f"  ✗ Quote FAIL — {e}")
    else:
        print("  \u26a0 Alpaca not configured (no API key/secret set)")
        print("    Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in .env or environment")

    # Liquid check (disabled by default)
    print("\n[3] Liquid Trade (live trading — Phase 2)...")
    liquid_cfg = config.get("liquid", {})
    if liquid_cfg.get("enabled", False):
        try:
            client = LiquidClient(enabled=True)
            status = client.status()
            print(f"  \u2713 OK — Liquid enabled (paper={status.get('paper', 'unknown')})")
        except Exception as e:
            print(f"  \u2717 FAIL — {e}")
    else:
        print("  \u2713 DISABLED — Liquid integration not active (paper trading via Alpaca only)")


def cmd_backfill(args, dp: DataProvider, config):
    """Backfill historical data for configured symbols."""
    symbols = resolve_market_universe(config, dp)
    period = config["yfinance"]["default_period"]
    interval = config["yfinance"]["default_interval"]

    print(f"=== Backfilling {len(symbols)} symbols ({period}/{interval}) ===")
    for symbol in symbols:
        df = dp.fetch_yf_history(symbol, period=period, interval=interval)
        if not df.empty:
            print(f"  ✓ {symbol}: {len(df)} rows, close: {df['Close'].iloc[-1]:.2f}")
        else:
            print(f"  ✗ {symbol}: failed")


def cmd_backtest(args, dp: DataProvider, config):
    """Run a simple SMA crossover backtest using yfinance historical data."""
    symbols = resolve_market_universe(config, dp)
    period = config["yfinance"]["default_period"]
    interval = config["yfinance"]["default_interval"]
    strategy, label = strategy_from_config(config)
    print(f"=== Backtest: {strategy.name} [{label}] (short={strategy.sma_short}, long={strategy.sma_long}) ===")
    print(f"Symbols: {symbols}, Period: {period}, Interval: {interval}\n")

    all_results = []
    for symbol in symbols:
        df = dp.fetch_yf_history(symbol, period=period, interval=interval)
        if df.empty:
            print(f"  ✗ {symbol}: no data, skipping")
            continue

        results = strategy.backtest(df)
        all_results.append((symbol, results))

        print(f"  {symbol}:")
        print(f"    Total Return:     {results['total_return_pct']:.2f}%")
        print(f"    Max Drawdown:     {results['max_drawdown_pct']:.2f}%")
        print(f"    Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
        print(f"    Total Trades:     {results['total_trades']}")
        print()

    # Summary
    print("=== Summary ===")
    for symbol, r in all_results:
        print(f"  {symbol}: {r['total_return_pct']:.2f}% return, {r['total_trades']} trades")


def cmd_account(args, dp: DataProvider, config):
    """Show Alpaca account state."""
    if not dp.trading_client:
        print("Alpaca not configured. Set API credentials to use this command.")
        return

    account = dp.get_account()
    print("=== Alpaca Account ===")
    for k, v in account.items():
        if v is not None and k in [
            "status", "cash", "portfolio_value", "currency",
            "pattern_day_trader", "trade_suspended_by_user",
            "created", "portfolio_value", "multiplier", "buying_power",
        ]:
            print(f"  {k}: {v}")

    positions = dp.get_positions()
    print(f"\n=== Positions ({len(positions)}) ===")
    if positions:
        for p in positions:
            print(f"  {p['symbol']}: {p['qty']} shares @ {p['avg_entry_price']} (unrealized P/L: {p.get('unrealized_pl', 'N/A')})")
    else:
        print("  No open positions.")


def cmd_run(args, dp: DataProvider, config):
    """
    One monitoring cycle: signals → research → advisor → Jev → risk.
    NEVER places orders (shows what `trade` would do).
    """
    _cycle(dp, config, execute=False)


def cmd_trade(args, dp: DataProvider, config):
    """One full cycle that places paper orders after Jev + risk approval."""
    _cycle(dp, config, execute=True)


def cmd_cycle(args, dp: DataProvider, config):
    """
    Scheduled entry point. Executes only when config `execution.auto_trade`
    is explicitly true (paper account required); otherwise monitor-only.
    Keeps monitoring and execution separate by explicit opt-in rather than
    by running two cron jobs.
    """
    execute = bool(config.get("execution", {}).get("auto_trade", False))
    _cycle(dp, config, execute=execute)


def etf_mode(config) -> bool:
    """strategy.mode: etf_trend (default since 2026-10-06) | stock_signals (legacy SMA-cross bot)."""
    return config.get("strategy", {}).get("mode", "stock_signals") == "etf_trend"


def _cycle(dp, config, execute: bool):
    if etf_mode(config):
        from engine_etf import run_etf_cycle
        return run_etf_cycle(dp, config, execute=execute)
    return run_cycle(dp, config, execute=execute)


def cmd_etf_status(args, dp: DataProvider, config):
    """Current ETF targets, per-ETF signal vs its 10-month average, and the saved plan."""
    from strategies import etf_trend as et
    from engine_etf import load_prices, STATE_PATH
    params = et.params_from_config(config)
    closes, _ = load_prices(dp, list(params["symbols"]) + [params["cash_symbol"]])
    targets, info = et.targets_asof(closes, None, params)
    print(f"{et.VERSION} — data through {info['asof']}, rebalance day: {info['rebalance_today']}")
    latest = {}
    for k, sig in info["signals"].items():
        for s, v in sig.items():
            latest.setdefault(s, v)
    for s, w in targets.sort_values(ascending=False).items():
        v = latest.get(s, {})
        extra = f"  price {v['pct_vs_avg']:+.1f}% vs 10-mo avg" if v else ""
        print(f"  {s:4s} target {w * 100:5.1f}%  in market {info['in_market'].get(s, '-')}/{info['tranches']}{extra}")
    from strategies import overlay as ov
    ovs = ov.settings(config.get("etf_trend") or {})
    if ov.active(ovs):
        final = ov.apply(targets, ovs, params["cash_symbol"])
        print(f"Overlay ON ({ov.label(config.get('etf_trend') or {}, et.VERSION)}): "
              f"{ovs['index_weight'] * 100:.0f}% {ovs['index_symbol']} + {(1 - ovs['index_weight']) * 100:.0f}% trend, "
              f"x{ovs['leverage']:g} → exposure {ov.gross(final, params['cash_symbol']) * 100:.0f}% of equity, "
              f"borrowing {ov.borrow_limit(final, params['cash_symbol']) * 100:.0f}%")
        print("  Final targets: " + ", ".join(f"{s} {w * 100:.1f}%" for s, w in final.sort_values(ascending=False).items()))
    st = json.load(open(STATE_PATH)) if os.path.exists(STATE_PATH) else {}
    print(f"State: {json.dumps({k: st.get(k) for k in ('status', 'strategy_version', 'gross_target', 'overlay_note', 'last_executed_decision', 'executed_at', 'pending_plan')}, default=str)}")


def cmd_etf_replay(args, dp: DataProvider, config):
    """Replay the LIVE decision code day by day over history against a simulated
    broker and compare with the strategy model (should track closely)."""
    import yfinance as yf
    from strategies import etf_trend as et
    from engine_etf import replay
    params = et.params_from_config(config)
    syms = list(params["symbols"]) + [params["cash_symbol"]]
    raw = yf.download(syms, period="max", auto_adjust=True, progress=False, group_by="ticker")
    closes = pd.DataFrame({s: raw[s]["Close"] for s in syms}).dropna()
    opens = pd.DataFrame({s: raw[s]["Open"] for s in syms}).reindex(closes.index)
    r = replay(closes, opens, args.start, config)
    yrs = len(r["replay"]) / 252
    cagr = lambda e: (e.iloc[-1] / e.iloc[0]) ** (1 / yrs) * 100 - 100
    print(f"Replay {r['replay'].index[0].date()} → {r['replay'].index[-1].date()} ({yrs:.1f} years, "
          f"{r['orders']} orders, whole shares, 0.10%/side)")
    print(f"  live code (replay): {r['replay_return_pct']:.1f}% total, {cagr(r['replay']):.2f}%/yr")
    print(f"  strategy model:     {r['model_return_pct']:.1f}% total, {cagr(r['model']):.2f}%/yr")
    print(f"  tracking error:     {r['tracking_error_pct']:.2f}% a year "
          f"({'OK' if r['tracking_error_pct'] < 1.5 else 'INVESTIGATE'})")


def cmd_review(args, dp: DataProvider, config):
    """Generate a daily review report with advisor + Jev decision trail."""
    from journal.audit_log import read_audit_log

    symbols = resolve_market_universe(config, dp)[:25]
    balance = config.get("paper_trading", {}).get("base_balance_usd", 100000)
    risk_mgr = RiskManager(config, initial_balance=balance)
    risk_mgr.load_state(os.path.join(os.path.dirname(__file__), "journal", "risk_state.json"))
    if dp.trading_client:
        try:
            risk_mgr.sync_from_broker(dp.get_account(), dp.get_positions())
        except Exception as e:
            logger.warning("Broker sync for review failed: %s", e)
    strategy, label = strategy_from_config(config)

    print(f"=== Daily Review ({datetime.now().isoformat()}) ===")
    print(f"Strategy: {strategy.name} [{label}]")
    print(f"Balance: ${balance:,.0f} (paper)\n")

    # Run backtest summary
    all_results = []
    for symbol in symbols:
        df = dp.fetch_yf_history(symbol, period="1y", interval="1d")
        if df.empty:
            continue
        results = strategy.backtest(df)
        all_results.append((symbol, results))

    print("=== Strategy Performance (1y backtest) ===")
    for symbol, r in all_results:
        print(f"  {symbol}: {r['total_return_pct']:.2f}% return, "
              f"{r['max_drawdown_pct']:.2f}% max DD, "
              f"{r['total_trades']} trades, Sharpe {r['sharpe_ratio']:.2f}")

    # Risk summary
    summary = risk_mgr.daily_summary()
    print(f"\n=== Risk Summary ===")
    print(f"  Daily P&L: ${summary['realized_pl']:.2f}")
    print(f"  Daily loss limit: ${summary['daily_loss_limit_usd']:.2f}")
    print(f"  Total exposure: ${summary['total_exposure']:.2f}")
    print(f"  Halted: {summary['halted']}")

    # Journal summary (original trade_log.csv)
    log_path = os.path.join(os.path.dirname(__file__), config["journal"]["log_file"])
    if os.path.exists(log_path):
        import csv
        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        trades = len([r for r in rows if r.get("status") in ("filled", "submitted", "simulated")])
        signals = len([r for r in rows if r.get("status") == "signal"])
        holds = len([r for r in rows if r.get("status") == "hold"])
        print(f"\n=== Journal ===")
        print(f"  Total logged events: {len(rows)}")
        print(f"  Executed trades: {trades}")
        print(f"  Signals (not executed): {signals}")
        print(f"  Hold decisions: {holds}")
    else:
        print(f"\n=== Journal ===")
        print("  No trades logged yet.")

    # Audit trail summary (new advisory decision flow)
    audit = read_audit_log()
    if audit:
        from collections import Counter
        outcomes = Counter(r.get("cycle_outcome", "unknown") for r in audit)
        print(f"\n=== Advisory Decision Audit ===")
        print(f"  Total events: {len(audit)}")
        for outcome, count in outcomes.most_common():
            print(f"  {outcome}: {count}")

        # Show advisor vs Jev agreement/disagreement
        adv_recs = [r.get("advisor_recommendation") for r in audit if r.get("advisor_recommendation")]
        jev_actions = [r.get("jev_action") for r in audit if r.get("jev_action")]
        disagreements = sum(
            1 for a, j in zip(adv_recs, jev_actions)
            if a and j and a != j
        )
        if adv_recs:
            print(f"  Advisor recommendations: {dict((r, adv_recs.count(r)) for r in set(adv_recs))}")
            print(f"  Jev decisions: {dict((a, jev_actions.count(a)) for a in set(jev_actions) if a)}")
            print(f"  Advisor vs Jev disagreements: {disagreements}")
    else:
        print(f"\n=== Advisory Decision Audit ===")
        print("  No audit events yet.")


def cmd_liquid_status(args, dp: DataProvider, config):
    """Check Liquid Trade integration status (disabled by default)."""
    liquid_cfg = config.get("liquid", {})
    client = LiquidClient(enabled=liquid_cfg.get("enabled", False))

    print("=== Liquid Trade Integration ===")
    status = client.status()
    print(f"  Enabled: {status['enabled']}")
    print(f"  Paper mode: {status['paper']}")
    if status["enabled"]:
        port = client.portfolio()
        print(f"  Portfolio: {port}")
    else:
        print(f"  → {status['message']}")
        print(f"  To enable: set liquid.enabled=true in config.yaml")
        print(f"  Requires: operator approval + paper-trading validation")
        print(f"  MCP URL: {client.mcp_url}")


def cmd_research(args, dp: DataProvider, config):
    """Run research roles on current snapshot for all symbols."""
    from research.roles import ResearchAggregator
    from engine import fetch_bars

    if not config.get("research", {}).get("enabled", False):
        print("Research roles are disabled in config.yaml")
        return

    symbols = resolve_market_universe(config, dp)
    strategy, _label = strategy_from_config(config)
    s = {"sma_short": strategy.sma_short, "sma_long": strategy.sma_long}
    bars, sources = fetch_bars(dp, symbols, 150)
    account = positions = None
    if dp.trading_client:
        try:
            account, positions = dp.get_account(), dp.get_positions()
        except Exception:
            pass

    aggregator = ResearchAggregator(config)
    aggregator.set_data_provider(dp)

    print(f"=== Research Analysis ({datetime.now().isoformat()}) ===")
    print(f"Roles: {config.get('research', {}).get('roles', [])}\n")

    for symbol in symbols:
        df = bars.get(symbol)
        source = sources.get(symbol, "?")
        if df is None or df.empty:
            print(f"  {symbol}: no data")
            continue

        strategy_signal = strategy.live_signal(df)
        snap = build_snapshot(
            dp=dp, risk_mgr=RiskManager(config, initial_balance=100000),
            symbol=symbol, strategy_signal=strategy_signal,
            strategy_name=strategy.name, df=df,
            sma_short=s["sma_short"], sma_long=s["sma_long"],
            account_state=account, positions=positions,
        )

        result = aggregator.run_full_analysis(snap, df)
        summary = result["summary"]

        print(f"  {symbol} [{source}] close={snap['market']['close']:.2f} signal={strategy_signal}")
        print(f"    Direction: {summary['overall_direction']} (bullish={summary['bullish']}, bearish={summary['bearish']}, neutral={summary['neutral']})")
        print(f"    Avg confidence: {summary['avg_confidence']:.2f}")

        for finding in result["findings"]:
            emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(finding["direction"], "⚪")
            print(f"    {emoji} [{finding['role']}] {finding['finding']} (conf={finding['confidence']:.2f})")
        print()


def cmd_learning_status(args, dp: DataProvider, config):
    """Show learning system status."""
    from learning.candidates import CandidateStore, CandidateStatus
    from learning.audit import LearningAuditLog

    store = CandidateStore(os.path.join(os.path.dirname(__file__), "learning", "candidates.csv"))
    audit = LearningAuditLog(os.path.join(os.path.dirname(__file__), "learning", "learning_audit_log.csv"))

    candidates = store.read_all()
    events = audit.read_all()

    print(f"=== Learning System Status ({datetime.now().isoformat()}) ===")
    print(f"  Total candidates: {len(candidates)}")
    print(f"  Audit events: {len(events)}")

    if candidates:
        from collections import Counter
        status_counts = Counter(c.status.value for c in candidates)
        print(f"\n  Candidate Status:")
        for status, count in status_counts.most_common():
            print(f"    {status}: {count}")

        print(f"\n  Recent Candidates:")
        for c in candidates[-5:]:
            print(f"    {c.candidate_id[:8]} v{c.version} [{c.status.value}] {c.hypothesis[:60]}")
    else:
        print("\n  No candidates yet. Use learning-evaluate to create and evaluate candidates.")

    if events:
        print(f"\n  Recent Audit Events:")
        for e in events[-5:]:
            print(f"    {e['timestamp'][:19]} {e['event_type']} {e.get('candidate_id', '')[:8]}")


def _learning_symbols(config, dp):
    lcfg = config.get("learning", {}).get("evaluation", {})
    syms = lcfg.get("symbols") or resolve_market_universe(config, dp)
    return list(syms)


def cmd_learn(args, dp: DataProvider, config):
    """
    Learning stage (outside the order path), run once per day after close:
      1. strategy search: propose → evaluate → shadow paper → promote
      2. decision-policy learning (does Jev's filtering help?)
      3. real-money readiness of the ACTUAL paper record
      4. Discord digest
    """
    from learning.loop import run_learning_cycle
    if not config.get("learning", {}).get("enabled", False):
        print("Learning system is disabled in config.yaml")
        return
    if etf_mode(config):
        return _learn_etf(args, dp, config)
    propose = getattr(args, "propose", None)
    path = run_learning_cycle(dp, config, _learning_symbols(config, dp), propose=propose)

    extra = []
    policy, rd = {}, {}
    try:
        from learning.decision_policy import update_policy
        from learning.forecasts import ForecastStore
        from engine import fetch_bars
        syms = sorted({f.symbol for f in ForecastStore(
            os.path.join(os.path.dirname(__file__), "learning", "forecasts.csv")).read_all()})
        bars, _ = fetch_bars(dp, syms, 200) if syms else ({}, {})
        policy = update_policy(config, bars)
        extra += ["## Decision policy (Jev / advisor value)",
                  f"- status: {policy.get('status')}, samples: {policy.get('samples')}, "
                  f"min_confidence_entry: {policy.get('min_confidence_entry', '-')}, "
                  f"jev_can_veto: {policy.get('jev_can_veto', '-')}",
                  f"- evidence: {json.dumps(policy.get('evidence', {}))[:600]}", ""]
    except Exception as e:
        extra += [f"_Decision policy update failed: {e}_", ""]
    try:
        from learning.readiness import assess_and_notify
        from engine import fetch_bars
        spy, _ = fetch_bars(dp, ["SPY"], 760)
        rd = assess_and_notify(config, spy["SPY"]["Close"] if "SPY" in spy else None)
        extra += ["## Real-money readiness (actual paper account)",
                  f"- **{rd['status']}** — {rd['summary']}",
                  f"- days {rd.get('days')}, closed trades {rd.get('closed_trades', 0)}, "
                  f"Sharpe {rd.get('sharpe', '-')}, max DD {rd.get('max_drawdown_pct', '-')}%, "
                  f"PSR {rd.get('psr', '-')}, SPY Sharpe {rd.get('benchmark_sharpe', '-')}", ""]
    except Exception as e:
        extra += [f"_Readiness assessment failed: {e}_", ""]

    with open(path, "a") as f:
        f.write("\n".join(extra) + "\n")
    with open(path) as f:
        report = f.read()
    print(report)
    print(f"Report: {path}")
    _daily_digest(config, dp, report, rd)


def _learn_etf(args, dp, config):
    """ETF mode: monitor (paper vs model vs benchmarks) + readiness + Jev shadow
    scoring. No strategy search: re-tuning made walk-forward results worse."""
    from learning.etf_monitor import report as etf_report
    path, _, rd = etf_report(dp, config)
    extra = []
    try:
        from learning.decision_policy import update_policy
        from learning.forecasts import ForecastStore
        from engine import fetch_bars
        syms = sorted({f.symbol for f in ForecastStore(
            os.path.join(os.path.dirname(__file__), "learning", "forecasts.csv")).read_all()})
        bars, _ = fetch_bars(dp, syms, 200) if syms else ({}, {})
        policy = update_policy(config, bars)
        extra += ["## Jev shadow scoring (never used for orders in ETF mode)",
                  f"- status: {policy.get('status')}, samples: {policy.get('samples')}, "
                  f"agreed-with avg {policy.get('evidence', {}).get('taken_mean_ret_pct', '-')}% vs "
                  f"would-have-blocked avg {policy.get('evidence', {}).get('vetoed_mean_ret_pct', '-')}%", ""]
    except Exception as e:
        extra += [f"_Jev shadow scoring failed: {e}_", ""]
    with open(path, "a") as f:
        f.write("\n".join(extra) + "\n")
    with open(path) as f:
        report = f.read()
    print(report)
    print(f"Report: {path}")
    _daily_digest(config, dp, report, rd)


def _daily_digest(config, dp, report: str, rd: dict):
    """Short end-of-day message to Discord."""
    import notify
    lines = ["📊 Paper bot today"]
    if dp.trading_client:
        try:
            a = dp.get_account()
            eq, last = float(a.get("equity") or 0), float(a.get("last_equity") or 0)
            pos = dp.get_positions()
            lines.append(f"💰 Account ${eq:,.0f} · today {eq - last:+,.0f}")
            lines.append(f"📦 {len(pos)} holdings")
        except Exception:
            lines.append("⚠️ Account unavailable. Check the dashboard.")
    lines.append("📌 ETFs" if etf_mode(config) else "📌 Stocks")
    if rd:
        lines.append("⚠️ Live trading: review needed" if rd.get("status") == "READY_FOR_REVIEW"
                     else "🔒 Paper only. Live trading not approved.")
    files = []
    try:
        from dashboard.render import write_dashboard, EQUITY_PNG_PATH
        acct = pos = None
        if dp.trading_client:
            try:
                acct, pos = dp.get_account(), dp.get_positions()
            except Exception:
                pass
        html_path = write_dashboard(config, acct, pos, png=True)
        files = [EQUITY_PNG_PATH, html_path]
        lines.append("📎 Chart attached · http://hermes-finbot/")
    except Exception:
        lines.append("⚠️ Dashboard unavailable.")
    # The legacy message remains the safe fallback when display-only inputs fail.
    legacy = "\n".join(lines)
    try:
        from eod_summary import build_summary, split_message
        acct = pos = None
        if dp.trading_client:
            acct, pos = dp.get_account(), dp.get_positions()
        spy_closes = None
        if hasattr(dp, "fetch_yf_history"):
            try:
                bars = dp.fetch_yf_history("SPY", period="2y", interval="1d")
                if bars is not None and len(bars) >= 2:
                    spy_closes = tuple(bars["Close"].iloc[-2:])
            except Exception:
                pass  # unknown is better than a stale or invented benchmark
        spy_history = ({str(index.date()): value for index, value in bars["Close"].items()}
                       if spy_closes is not None else None)
        text = build_summary(config, acct or {}, pos or [], rd or {}, spy_closes=spy_closes,
                             spy_history=spy_history)
        prefix = config.get("notifications", {}).get("prefix", "🤖 finance-bot")
        for part in split_message(text, prefix=prefix):
            notify.send(config, part, kind="daily", max_content_len=2000)
        legacy_parts = split_message(legacy, prefix=prefix)
        for index, part in enumerate(legacy_parts):
            notify.send(config, part, kind="daily", files=files if index == len(legacy_parts) - 1 else [], max_content_len=2000)
    except Exception as exc:
        logger.warning("EOD summary failed: %s", type(exc).__name__)
        from eod_summary import split_message
        prefix = config.get("notifications", {}).get("prefix", "🤖 finance-bot")
        parts = split_message(legacy + f"\n(summary unavailable: {type(exc).__name__})", prefix=prefix)
        for index, part in enumerate(parts):
            notify.send(config, part, kind="daily", files=files if index == len(parts) - 1 else [], max_content_len=2000)


def cmd_learning_evaluate(args, dp: DataProvider, config):
    """Walk-forward evaluation of existing candidates only (no new proposals)."""
    args.propose = 0
    cmd_learn(args, dp, config)


def cmd_learning_promote(args, dp: DataProvider, config):
    """Show promotion assessments from the stored learning audit trail."""
    from learning.audit import LearningAuditLog
    audit = LearningAuditLog(os.path.join(os.path.dirname(__file__), "learning", "learning_audit_log.csv"))
    events = [e for e in audit.read_all() if e["event_type"] in ("promotion_assessed", "evaluation_complete")]
    latest = {}
    for e in events:
        latest[e.get("candidate_id")] = e
    print(f"=== Promotion Assessment (latest per candidate) ===")
    if not latest:
        print("  No assessments yet. Run `python main.py learn` first.")
    for cid, e in latest.items():
        print(f"  {cid}: {e['event_type']} @ {e['timestamp'][:19]} — {e.get('rationale', '')[:200]}")


def cmd_learning_select(args, dp: DataProvider, config):
    """Model selection over STORED evaluation results."""
    from learning.candidates import CandidateStore, CandidateStatus
    from learning.loop import load_results, _selector, CANDIDATES_PATH
    store = CandidateStore(CANDIDATES_PATH)
    cands = [c for c in store.read_all() if c.status in (CandidateStatus.EVALUATING, CandidateStatus.PAPER)]
    result = _selector(config).select(cands, load_results())
    print(f"=== Model Selection ({datetime.now().isoformat()}) ===")
    print(f"  Candidates considered: {result.candidates_considered}, passing: {result.candidates_passing_gates}")
    print(f"  {result.selection_reason}")
    for sc in result.candidate_scores[:10]:
        print(f"    {sc['candidate_id']}: " + ", ".join(
            f"{k}={v:.2f}" for k, v in sc["scores"].items() if isinstance(v, (int, float))))


def cmd_learning_approve(args, dp: DataProvider, config):
    """Operator approval: make a candidate the active trading strategy."""
    from learning.loop import approve_candidate
    rec = approve_candidate(args.candidate_id, operator=os.getenv("USER", "operator"))
    print(f"Active strategy is now {rec['candidate_id']}: {rec['genome']}")
    print("Previous version kept for `python main.py learning-rollback`.")


def cmd_learning_veto(args, dp: DataProvider, config):
    """Cancel an announced (pending) automatic strategy switch."""
    from learning.loop import veto_pending
    cid = veto_pending(operator=os.getenv("USER", "operator"))
    print(f"Vetoed pending switch to {cid}" if cid else "No pending strategy switch.")
    if cid:
        import notify
        notify.send(config, f"🛑 Pending strategy switch to `{cid}` was vetoed.", kind="promotions")


def cmd_readiness(args, dp: DataProvider, config):
    """Real-money readiness of the actual paper record (never switches to live)."""
    if etf_mode(config):
        from learning.etf_monitor import report as etf_report
        _, _, rd = etf_report(dp, config)
        print(json.dumps(rd, indent=2, default=str))
        return
    from learning.readiness import assess
    from engine import fetch_bars
    spy, _ = fetch_bars(dp, ["SPY"], 760)
    rd = assess(config, spy["SPY"]["Close"] if "SPY" in spy else None)
    print(json.dumps(rd, indent=2, default=str))


def cmd_learning_rollback(args, dp: DataProvider, config):
    from learning.loop import rollback_active
    prev = rollback_active(operator=os.getenv("USER", "operator"))
    print(f"Rolled back. Active: {prev['candidate_id'] if prev else 'config.yaml strategy'}")


def cmd_dashboard(args, dp: DataProvider, config):
    """
    Live web dashboard (same page as reports/dashboard.html, refreshed per request).

    Local:   python main.py dashboard                → http://127.0.0.1:8080
    Remote:  python main.py dashboard --host 0.0.0.0 (e.g. over Tailscale) —
             requires DASHBOARD_TOKEN; open http://<host>:8080/?token=<token>
    """
    try:
        from dashboard import create_app
    except ImportError as e:
        print(f"Dashboard not available: {e}  (pip install flask)")
        return
    host, port = args.host, args.port
    token = os.getenv("DASHBOARD_TOKEN")
    if host not in ("127.0.0.1", "localhost") and not token:
        print("Refusing to listen beyond localhost without DASHBOARD_TOKEN set (/opt/data/.env).")
        return
    app = create_app(config, dp=dp, token=token if host not in ("127.0.0.1", "localhost") else None)
    print(f"=== Dashboard on http://{host}:{port} ===")
    app.run(host=host, port=port, debug=False)


def cmd_forecast_record(args, dp: DataProvider, config):
    """Record a new forecast (append-only, immutable)."""
    from learning.forecasts import Forecast, ForecastStore, generate_forecast_id

    symbol = args.symbol
    horizon = args.horizon
    expected = args.expected
    confidence = args.confidence
    thesis = args.thesis
    strategy_version = args.strategy_version or strategy_from_config(config)[1]

    decision_time = datetime.now(timezone.utc).isoformat()
    forecast_id = generate_forecast_id(symbol, decision_time, thesis)

    forecast = Forecast(
        forecast_id=forecast_id,
        decision_time=decision_time,
        symbol=symbol,
        horizon_days=horizon,
        expected_return_pct=expected,
        confidence=confidence,
        thesis=thesis,
        source_publication_times="[]",
        strategy_version=strategy_version,
        data_snapshot_id="",
    )

    store = ForecastStore(os.path.join(os.path.dirname(__file__), "learning", "forecasts.csv"))
    store.record(forecast)

    print(f"=== Forecast Recorded ===")
    print(f"  ID: {forecast.forecast_id}")
    print(f"  Symbol: {forecast.symbol}")
    print(f"  Horizon: {forecast.horizon_days} days")
    print(f"  Expected: {forecast.expected_return_pct:.2f}%")
    print(f"  Confidence: {forecast.confidence:.2f}")
    print(f"  Strategy: {forecast.strategy_version}")
    print(f"  Status: {forecast.status.value}")


def cmd_forecast_evaluate(args, dp: DataProvider, config):
    """Evaluate all matured forecasts."""
    from learning.forecasts import ForecastStore, ForecastStatus
    from learning.forecast_evaluator import ForecastEvaluator

    store = ForecastStore(os.path.join(os.path.dirname(__file__), "learning", "forecasts.csv"))
    evaluator = ForecastEvaluator()

    matured = store.get_matured()
    if not matured:
        print("=== Forecast Evaluation ===")
        print("  No matured forecasts to evaluate.")
        return

    # Fetch price data for each symbol
    price_data = {}
    for f in matured:
        if f.symbol not in price_data:
            df = pd.DataFrame()
            if dp.data_client:
                try:
                    end = datetime.now()
                    start = end - timedelta(days=365)
                    df = dp.fetch_alpaca_bars(f.symbol, start, end)
                except Exception:
                    pass
            if df.empty:
                df = dp.fetch_yf_history(f.symbol, period="1y", interval="1d")
            price_data[f.symbol] = df

    batch = evaluator.evaluate_batch(matured, price_data)

    print(f"=== Forecast Evaluation ({datetime.now(timezone.utc).isoformat()}) ===")
    print(f"  Total evaluated: {batch.total_evaluated}")
    print(f"  Avg forecast error: {batch.avg_forecast_error:.2f}%")
    print(f"  Avg calibration error: {batch.avg_calibration_error:.4f}")
    print(f"  Avg net return: {batch.avg_net_return:.2f}%")
    print(f"  Hit rate: {batch.hit_rate:.1%}")

    if batch.calibration_bins:
        print(f"\n  Calibration Bins:")
        for b in batch.calibration_bins:
            if b.count > 0:
                print(f"    {b.bin_start:.1f}-{b.bin_end:.1f}: n={b.count}, "
                      f"conf={b.avg_confidence:.2f}, actual={b.avg_actual:.2f}, "
                      f"error={b.calibration_error:.2f}")

    # Update statuses
    for f in matured:
        store.update_status(f.forecast_id, ForecastStatus.EVALUATED)


def cmd_forecast_status(args, dp: DataProvider, config):
    """Show forecast status and calibration summary."""
    from learning.forecasts import ForecastStore

    store = ForecastStore(os.path.join(os.path.dirname(__file__), "learning", "forecasts.csv"))
    stats = store.get_stats()

    print(f"=== Forecast Status ({datetime.now(timezone.utc).isoformat()}) ===")
    print(f"  Total forecasts: {stats['total']}")
    print(f"  By status: {stats['by_status']}")
    print(f"  By symbol: {stats['by_symbol']}")


def cmd_challenger_compare(args, dp: DataProvider, config):
    """Compare challenger strategies against active and baselines."""
    from learning.candidates import CandidateStore, CandidateConfig, Candidate, CandidateStatus
    from learning.challenger_comparison import ChallengerComparison, ComparisonConfig
    from learning.audit import LearningAuditLog
    from strategies.v1_1 import SMACrossV1_1

    challenger_id = args.challenger_id

    store = CandidateStore(os.path.join(os.path.dirname(__file__), "learning", "candidates.csv"))
    audit = LearningAuditLog(os.path.join(os.path.dirname(__file__), "learning", "learning_audit_log.csv"))

    candidate = store.get(challenger_id)
    if candidate is None:
        print(f"Candidate {challenger_id} not found.")
        return

    # Build strategies
    challenger_strategy = SMACrossV1_1(
        sma_short=candidate.config.sma_short,
        sma_long=candidate.config.sma_long,
        atr_period=candidate.config.atr_period,
        atr_mult_initial=candidate.config.atr_mult_initial,
    )

    active_strategy, _label = strategy_from_config(config)

    comp_config = ComparisonConfig()
    comparison = ChallengerComparison(comp_config)

    symbols = resolve_market_universe(config, dp)

    print(f"=== Challenger Comparison ({datetime.now(timezone.utc).isoformat()}) ===")
    print(f"  Challenger: {candidate.candidate_id} v{candidate.version}")
    print(f"  Active: {active_strategy.name} v{active_strategy.version}")
    print(f"  Symbols: {symbols}")
    print()

    for symbol in symbols:
        df = pd.DataFrame()
        if dp.data_client:
            try:
                end = datetime.now()
                start = end - timedelta(days=730)
                df = dp.fetch_alpaca_bars(symbol, start, end)
            except Exception:
                pass
        if df.empty:
            df = dp.fetch_yf_history(symbol, period="2y", interval="1d")

        if df.empty:
            print(f"  {symbol}: no data, skipping")
            continue

        results = comparison.compare_all_periods(
            challenger=candidate,
            challenger_strategy=challenger_strategy,
            active_strategy=active_strategy,
            df=df,
            symbol=symbol,
        )

        for r in results:
            status = "PASS" if r.challenger_passes_gates else "FAIL"
            print(f"  {symbol} [{r.period}]: {status}")
            print(f"    Challenger: return={r.challenger_return_pct:.2f}%, "
                  f"sharpe={r.challenger_sharpe:.2f}, "
                  f"dd={r.challenger_max_drawdown_pct:.2f}%, "
                  f"trades={r.challenger_trades}")
            print(f"    Active:     return={r.active_return_pct:.2f}%, "
                  f"sharpe={r.active_sharpe:.2f}, "
                  f"dd={r.active_max_drawdown_pct:.2f}%, "
                  f"trades={r.active_trades}")
            print(f"    Baselines:  SMA={r.baseline_sma_sharpe:.2f}, "
                  f"B&H={r.baseline_buyhold_sharpe:.2f}")
            if r.rejection_reasons:
                print(f"    Rejected: {'; '.join(r.rejection_reasons[:2])}")

        # Log to audit
        audit.log_event(
            "challenger_compared",
            candidate.candidate_id,
            candidate.version,
            details={"symbol": symbol, "results": [r.to_dict() for r in results]},
            rationale="Challenger comparison completed",
        )


def main():
    parser = argparse.ArgumentParser(description="Combined Trading Bot (Alpaca + yfinance / Liquid)")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Simple commands (no extra args)
    for cmd in ["health", "backfill", "backtest", "run", "trade", "cycle", "account",
                "review", "liquid-status", "research", "learning-status",
                "learning-evaluate", "learning-promote", "learning-select",
                "learning-rollback", "learning-veto", "readiness", "forecast-evaluate", "forecast-status",
                "etf-status"]:
        subparsers.add_parser(cmd)

    er = subparsers.add_parser("etf-replay")
    er.add_argument("--start", default="2010-01-01", help="first replay day (needs ~1 year of prior data)")

    ln = subparsers.add_parser("learn")
    ln.add_argument("--propose", type=int, default=None,
                    help="new candidates to propose this run (default: learning.propose_per_run)")

    db = subparsers.add_parser("dashboard")
    db.add_argument("--host", default="127.0.0.1")
    db.add_argument("--port", type=int, default=8080)

    la = subparsers.add_parser("learning-approve")
    la.add_argument("--candidate-id", required=True)

    # forecast-record
    fr = subparsers.add_parser("forecast-record")
    fr.add_argument("--symbol", required=True)
    fr.add_argument("--horizon", type=int, required=True)
    fr.add_argument("--expected", type=float, required=True)
    fr.add_argument("--confidence", type=float, required=True)
    fr.add_argument("--thesis", required=True)
    fr.add_argument("--strategy-version", default=None)

    # challenger-compare
    cc = subparsers.add_parser("challenger-compare")
    cc.add_argument("--challenger-id", required=True)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    config = load_config()

    paper = os.getenv("APCA_PAPER", "true").lower() == "true"
    dp = DataProvider(paper=paper)

    # Ensure journal and audit log files are initialized
    ensure_header()
    ensure_audit_header()

    commands = {
        "health": cmd_health,
        "backfill": cmd_backfill,
        "backtest": cmd_backtest,
        "run": cmd_run,
        "trade": cmd_trade,
        "cycle": cmd_cycle,
        "learn": cmd_learn,
        "learning-approve": cmd_learning_approve,
        "learning-rollback": cmd_learning_rollback,
        "learning-veto": cmd_learning_veto,
        "readiness": cmd_readiness,
        "account": cmd_account,
        "review": cmd_review,
        "liquid-status": cmd_liquid_status,
        "research": cmd_research,
        "learning-status": cmd_learning_status,
        "learning-evaluate": cmd_learning_evaluate,
        "learning-promote": cmd_learning_promote,
        "learning-select": cmd_learning_select,
        "dashboard": cmd_dashboard,
        "forecast-record": cmd_forecast_record,
        "forecast-evaluate": cmd_forecast_evaluate,
        "forecast-status": cmd_forecast_status,
        "challenger-compare": cmd_challenger_compare,
        "etf-status": cmd_etf_status,
        "etf-replay": cmd_etf_replay,
    }

    commands[args.command](args, dp, config)


if __name__ == "__main__":
    main()
