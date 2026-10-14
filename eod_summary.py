"""Read-only plain-English paper-account EOD view; no broker or order calls."""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
PACIFIC = ZoneInfo("America/Los_Angeles")


def _number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def _money(value):
    n = _number(value)
    return f"${n:,.0f}" if n is not None else "n/a"


def _pct(value):
    n = _number(value)
    return f"{n:+.1f}%" if n is not None else "n/a"


def _rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _today(value, today):
    if not value:
        return False
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            return timestamp.date() == today
        return timestamp.astimezone(PACIFIC).date() == today
    except ValueError:
        return False


def _ai_report_lines(config, journal, today):
    """Summarize the recorded AI decision; never call the model or change state."""
    if not config.get("etf_trend", {}).get("ai", {}).get("enabled"):
        return []
    path = journal / "ai_pm.json"
    try:
        records = json.loads(path.read_text()).get("decisions", []) if path.exists() else []
    except (OSError, ValueError, TypeError):
        records = []
    current = [r for r in records if isinstance(r, dict) and _today(r.get("decided_at"), today)]
    if not current:
        return ["", "AI portfolio manager", "No new AI decision recorded today."]
    rec = current[-1]
    plan = rec.get("proposal") if isinstance(rec.get("proposal"), dict) else {}
    status = rec.get("status") if rec.get("status") in ("accepted", "pending", "approved", "declined", "fallback") else "n/a"
    summary = " ".join(str(plan.get("summary") or "No summary provided.").split())[:280].replace("@", "＠")
    tilts = plan.get("tilts") if isinstance(plan.get("tilts"), dict) else {}
    changes = ", ".join(f"{symbol} {n * 100:+.0f}%" for symbol, value in sorted(tilts.items())
                         if isinstance(symbol, str) and symbol.isalnum() and len(symbol) <= 8
                         if (n := _number(value)) is not None) or "none"
    confidence = _number(plan.get("confidence"))
    confidence_text = f"{confidence * 100:.0f}%" if confidence is not None else "n/a"
    asof = rec.get("asof") if isinstance(rec.get("asof"), str) and len(rec["asof"]) == 10 else "n/a"
    return ["", "AI portfolio manager",
            f"Today's decision (data through {asof}; {status}): {summary}",
            f"Proposed tilts: {changes} · confidence {confidence_text}."]


def split_message(text, prefix="🤖 finance-bot", limit=2000):
    """Return lossless chunks, reserving space for notify.send's prefix."""
    capacity = limit - len(f"**{str(prefix)[:80]}** — ")
    if capacity < 1:
        raise ValueError("notification prefix leaves no message capacity")
    return [text[i:i + capacity] for i in range(0, len(text), capacity)] or [""]


def build_summary(config, account, positions, rd, *, base=BASE, now=None, spy_closes=None, spy_history=None):
    """Build from copy-local journals and supplied read-only broker snapshot."""
    now = now or datetime.now(PACIFIC)
    today = now.astimezone(PACIFIC).date()
    base = Path(base)
    journal = base / "journal"
    state_path = journal / "etf_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    rows = _rows(journal / "equity_curve.csv")
    trades = [r for r in _rows(journal / "trade_log.csv") if _today(r.get("timestamp"), today)]
    fills = {r.get("order_id"): r for r in _rows(journal / "etf_fills.csv")
             if r.get("order_id") and r.get("status", "").lower() == "filled"}
    eq = _number(account.get("equity") or account.get("portfolio_value"))
    last = _number(account.get("last_equity"))
    delta = eq - last if eq is not None and last is not None else None
    change = delta / last * 100 if delta is not None and last else None
    spy_day = None
    if spy_closes and len(spy_closes) == 2:
        prior, current = map(_number, spy_closes)
        if prior and current is not None:
            spy_day = (current / prior - 1) * 100
    label = state.get("strategy_version")
    if not label and config.get("strategy", {}).get("mode") == "etf_trend":
        from learning.etf_monitor import config_label
        label = config_label(config)
    track = [r for r in rows if label and r.get("strategy_version") == label
             and _number(r.get("equity")) is not None and r.get("date", "") <= today.isoformat()]
    track.sort(key=lambda r: r.get("date", ""))
    bot_return = (_number(track[-1]["equity"]) / _number(track[0]["equity"]) - 1) * 100 if len(track) > 1 and _number(track[0]["equity"]) else None
    spy_return = None
    if len(track) > 1 and spy_history is not None:
        start, end = track[0]["date"], track[-1]["date"]
        start_px, end_px = _number(spy_history.get(start)), _number(spy_history.get(end))
        if start_px and end_px:
            spy_return = (end_px / start_px - 1) * 100
    status = str(state.get("status") or "") if _today(state.get("updated_at"), today) else ""
    blocked = status.lower().startswith(("blocked", "waiting", "executed with"))
    alerts = [status] if blocked else []
    for event in _rows(journal / "audit_log.csv"):
        if not _today(event.get("timestamp"), today):
            continue
        outcome = event.get("cycle_outcome") or ""
        if outcome.endswith("failed") or outcome in ("risk_rejected", "manual_review") or event.get("execution_error"):
            alerts.append(f"{outcome or 'error'}: {event.get('execution_error') or event.get('advisor_error') or event.get('jev_error') or 'reason n/a'}")
    if alerts:
        opening = f"⚠️ Problem day — {('Trading paused: ' if blocked else '')}{alerts[0]}."
    elif delta is None:
        opening = "Today's account move is n/a."
    elif delta == 0:
        opening = "Quiet day — No change in paper account value today."
    elif change is not None and abs(change) < 0.5:
        opening = f"Quiet day — The paper account {'rose' if delta > 0 else 'fell'} {_money(abs(delta))} today."
    elif delta > 0:
        opening = f"{'Big up' if change is not None and change > 2 else 'Up'} day — The paper account rose {_money(delta)} today."
    elif delta < 0:
        opening = f"{'Big down' if change is not None and change < -2 else 'Down'} day — The paper account fell {_money(abs(delta))} today."

    activity = f"{len(trades)} order(s) logged today." if trades else "No trades logged today."
    brief = [opening, activity, "Paper trading only; live trading is not approved."]
    if alerts:
        brief[1] = "No new trading should be assumed from this status; check the order log below."

    values = [_number(r.get("equity")) for r in track]
    peak = drawdown = None
    for value in values:
        peak = max(peak or value, value)
        if peak > 0:
            drawdown = max(drawdown or 0, (peak - value) / peak * 100)
    dd = rd.get("max_drawdown_pct") if rd.get("max_drawdown_pct") is not None else (drawdown if len(values) > 1 else None)
    lines = ["📊 Paper bot today", "📝 Today in brief", " ".join(brief), "", "Performance",
             f"Equity {_money(eq)} · day {('+' if delta >= 0 else '-') + _money(abs(delta)) if delta is not None else 'n/a'} ({_pct(change)}).",
             f"SPY day {_pct(spy_day)} · Bot vs SPY since strategy label {label or 'n/a'}: "
             f"{_pct(bot_return)} vs {_pct(spy_return)} (same start/end dates; SPY historical closes).",
             f"Max drawdown {_pct(dd)}.",
             "", "Trades today"]
    if trades:
        for trade in trades:
            fill = fills.get(trade.get("order_id"))
            px = _number(fill.get("filled_avg_price")) if fill else None
            slip = _number(fill.get("slippage_bp")) if fill else None
            price_text = (f"fill {_money(px)}" if px is not None
                          else f"planned {_money(trade.get('price'))}, fill n/a")
            slip_text = f"{slip:.1f} bp" if slip is not None else "n/a"
            lines.append(f"{trade.get('action', 'n/a').capitalize()} {trade.get('qty') or 'n/a'} "
                         f"{trade.get('symbol') or 'n/a'}: {price_text} · slippage {slip_text} · status {trade.get('status') or 'n/a'}")
    else:
        lines.append("No trades logged today.")
    lines += ["", "Holdings"]
    def position_value(p):
        value = _number(p.get("market_value"))
        if value is None:
            qty, price = _number(p.get("qty")), _number(p.get("current_price"))
            value = qty * price if qty is not None and price is not None else None
        return value
    ordered = sorted(positions or [], key=lambda p: -(abs(position_value(p)) if position_value(p) is not None else -1))
    exposure = 0.0
    complete = bool(ordered)
    for p in ordered:
        value = position_value(p)
        if value is None:
            complete = False
        else:
            exposure += abs(value)
        lines.append(f"{p.get('symbol') or 'n/a'} {_money(value)} "
                     f"({value / eq * 100:.1f}% of equity)" if value is not None and eq else
                     f"{p.get('symbol') or 'n/a'} {_money(value)} (weight n/a)")
    if not ordered:
        lines.append("No holdings reported.")
    cash = _number(account.get("cash"))
    lines.append(f"Exposure {exposure / eq * 100:.1f}%" if complete and eq else "Exposure n/a")
    lines.append(f"Borrowed {_money(-cash if cash < 0 else 0) if cash is not None else 'n/a'}")
    lines += ["", "Trend signals"]
    signals = state.get("signals") or {}
    latest = state.get("tranche_last") or {}
    symbols = config.get("etf_trend", {}).get("symbols") or list((state.get("in_market") or {}).keys())
    if not symbols:
        lines.append("n/a (no saved strategy signals).")
    for symbol in sorted(symbols):
        choices = [(latest.get(str(k)) or "", v[symbol]) for k, v in signals.items() if symbol in v]
        sig = max(choices, key=lambda item: item[0])[1] if choices else {}
        distance = _number(sig.get("pct_vs_avg"))
        active = sig.get("in")
        if distance is None:
            lines.append(f"{symbol}: n/a")
        else:
            threshold = -1 if active else 1
            watch = " · Watch: within 1% of flip threshold" if abs(distance - threshold) <= 1 else ""
            lines.append(f"{symbol}: {'in trend' if active else 'out of trend'}; "
                         f"{distance:+.1f}% vs 10-month average{watch}")
    days = rd.get("days") if rd.get("days") is not None else (len(track) if track else None)
    min_days = (rd.get("targets") or {}).get("min_days", 252)
    next_day = "n/a"
    try:
        import pandas as pd
        asof = max(pd.Timestamp(state["asof"]), pd.Timestamp(today))
        tranche_days = config.get("etf_trend", {}).get("tranche_days", [0, 5, 10, 15])
        for day in pd.bdate_range(asof + pd.Timedelta(days=1), periods=35):
            month = pd.bdate_range(day.replace(day=1), day)
            if len(month) - 1 in tranche_days:
                next_day = f"tranche {tranche_days.index(len(month) - 1) + 1} on {day.date().isoformat()} (weekday estimate; holidays may shift it)"
                break
    except (KeyError, ValueError, TypeError):
        pass
    lines += _ai_report_lines(config, journal, today)
    lines += ["", "Next up", f"Next {next_day}. Readiness clock {days if days is not None else 'n/a'}/{min_days} trading days.",
              f"Tracking error {_pct(rd.get('tracking_error_pct'))} vs model; "
              f"drawdown {_pct(dd)} vs 20% readiness limit"
              + (f" (separate {config['etf_trend']['max_drawdown_halt_pct']:g}% trading halt)."
                 if config.get("etf_trend", {}).get("max_drawdown_halt_pct") is not None else "."),
              "", "Alerts today", "\n".join(alerts) if alerts else "None."]
    return "\n".join(lines)
