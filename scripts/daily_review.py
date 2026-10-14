#!/usr/bin/env python3
"""
Cron daily-review script for the combined trading bot.

Generates one dated report under /opt/data/finance-bot/daily_reviews/
and logs a terse status. Intended to run at 13:30 Pacific each weekday.

Usage (cron):
  30 20,21 * * 1-5 /opt/data/finance-bot/.venv/bin/python /opt/data/finance-bot/scripts/daily_review.py
"""
import os
import sys
import yaml
from datetime import datetime
from collections import Counter

BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BOT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BOT_DIR, ".env"), override=False)
load_dotenv(os.getenv("HERMES_ENV_FILE", "/opt/data/.env"), override=False)

from data import DataProvider
from risk import RiskManager
from journal.trade_log import HEADERS
from journal.audit_log import read_audit_log
from engine import resolve_market_universe, strategy_from_config, fetch_bars, read_closed_trades
import csv


def generate_review():
    config_path = os.path.join(BOT_DIR, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    paper = os.getenv("APCA_PAPER", "true").lower() == "true"
    dp = DataProvider(paper=paper)
    balance = config.get("paper_trading", {}).get("base_balance_usd", 100000)
    risk_mgr = RiskManager(config, initial_balance=balance)
    risk_mgr.load_state(os.path.join(BOT_DIR, "journal", "risk_state.json"))
    if dp.trading_client:
        try:
            risk_mgr.sync_from_broker(dp.get_account(), dp.get_positions())
        except Exception as e:
            print(f"Broker sync failed: {e}")
    strategy, strat_label = strategy_from_config(config)
    symbols = resolve_market_universe(config, dp)[:25]

    # Collect backtest results (batched bars, ~1y)
    results = []
    bars, _ = fetch_bars(dp, symbols, 380)
    for symbol in symbols:
        df = bars.get(symbol)
        if df is None or df.empty:
            continue
        r = strategy.backtest(df)
        results.append((symbol, r))

    # Read journal
    log_path = os.path.join(BOT_DIR, config["journal"]["log_file"])
    journal_rows = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            reader = csv.DictReader(f)
            journal_rows = list(reader)

    # Read audit log
    audit_rows = read_audit_log()

    # Build report
    today = datetime.now().strftime("%Y-%m-%d")
    review_dir = os.path.join(BOT_DIR, "daily_reviews")
    os.makedirs(review_dir, exist_ok=True)
    report_path = os.path.join(review_dir, f"{today}_review.md")

    lines = [f"# Daily Review — {today}", "",
             f"**Strategy:** {strategy.name} [{strat_label}]",
             f"**Balance:** ${balance:,.0f} (paper)", ""]

    lines.append("## Strategy Performance (1y backtest)")
    lines.append("")
    lines.append("| Symbol | Return | Max Drawdown | Sharpe | Trades |")
    lines.append("|--------|--------|-------------|--------|--------|")
    for sym, r in results:
        lines.append(f"| {sym} | {r['total_return_pct']:.2f}% | {r['max_drawdown_pct']:.2f}% | {r['sharpe_ratio']:.2f} | {r['total_trades']} |")
    lines.append("")

    summary = risk_mgr.daily_summary()
    lines.append("## Risk Summary")
    lines.append(f"- Equity: ${summary['current_balance']:,.2f} (peak ${summary['peak_balance']:,.2f})")
    lines.append(f"- Day P&L: ${summary['realized_pl']:.2f}")
    lines.append(f"- Total exposure: ${summary['total_exposure']:.2f}")
    lines.append(f"- Halted: {summary['halted']}")
    lines.append("")

    trades = len([r for r in journal_rows if r.get("status") in ("filled", "submitted", "simulated")])
    signals = len([r for r in journal_rows if r.get("status") == "signal"])
    holds = len([r for r in journal_rows if r.get("status") == "hold"])
    lines.append("## Journal")
    lines.append(f"- Executed trades: {trades}")
    lines.append(f"- Signal events logged: {signals}")
    lines.append(f"- Hold decisions: {holds}")
    lines.append("")

    # Advisory decision audit trail
    lines.append("## Advisory Decision Audit")
    if audit_rows:
        outcomes = Counter(r.get("cycle_outcome", "unknown") for r in audit_rows)
        adv_cfg = config.get("advisor", {})
        jev_cfg = config.get("jev", {})
        lines.append(f"- Total events: {len(audit_rows)}")
        lines.append(f"- Advisor: {adv_cfg.get('provider')} / {adv_cfg.get('model')} "
                      f"(enabled={adv_cfg.get('enabled', False)})")
        lines.append(f"- Jev: {jev_cfg.get('provider')} / {jev_cfg.get('model')} "
                      f"(enabled={jev_cfg.get('enabled', False)})")
        lines.append("")

        lines.append("| Outcome | Count |")
        lines.append("|---------|-------|")
        for outcome, count in outcomes.most_common():
            lines.append(f"| {outcome} | {count} |")
        lines.append("")

        # Advisor vs Jev agreement table
        adv_recs = [r.get("advisor_recommendation") for r in audit_rows if r.get("advisor_recommendation")]
        jev_actions = [r.get("jev_action") for r in audit_rows if r.get("jev_action")]
        disagreements = sum(1 for a, j in zip(adv_recs, jev_actions) if a and j and a != j)

        lines.append("### Advisor vs Jev Agreement")
        lines.append("")
        lines.append(f"- Advisor recommendations: {dict(Counter(adv_recs)) if adv_recs else 'none'}")
        lines.append(f"- Jev decisions: {dict(Counter(j for j in jev_actions if j)) if jev_actions else 'none'}")
        lines.append(f"- Disagreements (advisor advice overridden by Jev): {disagreements}")
        lines.append("")

        # Recent trades with advisor + Jev detail
        filled = [r for r in audit_rows if r.get("execution_status") in ("filled", "submitted")]
        if filled:
            lines.append("### Recent Filled Trades (advisor advice → Jev decision → execution)")
            lines.append("")
            lines.append("| Symbol | Advisor Advice | Advisor Conf | Jev Action | Jev Conf | Risk | Order ID | Price |")
            lines.append("|--------|----------------|-------------|------------|----------|------|----------|-------|")
            for r in filled[-20:]:
                lines.append(
                    f"| {r.get('symbol','')} | {r.get('advisor_recommendation','')} "
                    f"| {r.get('advisor_confidence','')} | {r.get('jev_action','')} "
                    f"| {r.get('jev_confidence','')} | {r.get('risk_action','')} "
                    f"| {r.get('order_id','')} | {r.get('execution_price','')} |"
                )
            lines.append("")

        # Risk rejections and skips
        rejected = [r for r in audit_rows if r.get("cycle_outcome") == "risk_rejected"]
        if rejected:
            lines.append("### Risk Rejections")
            lines.append("")
            for r in rejected[-10:]:
                lines.append(f"- {r.get('symbol','')}: Jev={r.get('jev_action','')} → "
                            f"Risk {r.get('risk_action','')}: {r.get('risk_reason','')[:80]}")
            lines.append("")

        # Skips (advisor/jev failures)
        skips = [r for r in audit_rows if r.get("cycle_outcome") in ("advisor_failed", "jev_failed")]
        if skips:
            lines.append("### Skipped Cycles (advisor/Jev failure)")
            lines.append("")
            for r in skips[-10:]:
                lines.append(f"- {r.get('symbol','')}: {r.get('cycle_outcome','')}")
            lines.append("")
        closed = read_closed_trades()
        if closed:
            pnl = [float(r["pnl_pct"]) for r in closed if r.get("pnl_pct") not in ("", None)]
            wins = [x for x in pnl if x > 0]
            lines.append("### Closed Trades (learning feedback)")
            lines.append("")
            lines.append(f"- Count: {len(closed)}, win rate: {len(wins)/len(pnl)*100:.0f}%" if pnl else f"- Count: {len(closed)}")
            if pnl:
                lines.append(f"- Avg return per trade: {sum(pnl)/len(pnl):.2f}%")
            for r in closed[-10:]:
                lines.append(f"- {r['closed_at'][:10]} {r['symbol']} {r['pnl_pct']}% ({r['exit_reason']}, {r['strategy_version']})")
            lines.append("")
    else:
        lines.append("- No audit events recorded.")
        lines.append("- Run `python main.py trade` to generate advisory decision events.")
        lines.append("")
    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report written: {report_path}")
    print(f"Summary: {len(results)} symbols backtested, {trades} trades, {signals} signals")
    if audit_rows:
        outcomes = Counter(r.get("cycle_outcome", "unknown") for r in audit_rows)
        print(f"Audit: {len(audit_rows)} events — {dict(outcomes)}")
    return report_path


if __name__ == "__main__":
    generate_review()