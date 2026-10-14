"""
HTML dashboard: Tabler (layout) + TradingView Lightweight Charts, both from
jsDelivr with pinned versions and SRI hashes. If the CDN is unreachable the
page still renders its data, and the charts fall back to inline SVG.

Written to reports/dashboard.html after every cycle and end-of-day run, so it
reaches Google Drive through the existing Hermes drive sync, can be attached
to Discord, and is served live by the Flask app (`python main.py dashboard`).

Also renders reports/equity.png (account vs SPY + drawdown) for the Discord
daily summary when matplotlib is installed.
"""
from __future__ import annotations

import csv
import html
import json
import os
from datetime import datetime, timezone, timedelta, date
from zoneinfo import ZoneInfo
from typing import Optional

BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BOT_DIR, "reports")
DASHBOARD_PATH = os.path.join(REPORTS_DIR, "dashboard.html")
EQUITY_PNG_PATH = os.path.join(REPORTS_DIR, "equity.png")
SPY_CACHE = os.path.join(BOT_DIR, "journal", "cache", "spy_close.json")
CALENDAR_CACHE = os.path.join(BOT_DIR, "journal", "cache", "etf_calendar.json")

# Validated two-series palette (reference categorical slots 1–2)
ACCOUNT = {"light": "#2a78d6", "dark": "#3987e5"}
SPY = {"light": "#eb6834", "dark": "#d95926"}
ASSET_CLASS = {"SPY": "US stocks", "IWM": "US small caps", "EFA": "International developed",
               "EEM": "Emerging markets", "IEF": "Treasuries 7-10y", "TLT": "Treasuries 20y+",
               "GLD": "Gold", "DBC": "Commodities", "VNQ": "Real estate", "BIL": "T-bills (cash)"}


def _p(*parts):
    return os.path.join(BOT_DIR, *parts)


def _csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def spy_closes(start: str) -> dict:
    """{date: close} for SPY since `start` (incl. the prior close for as-of
    alignment), cached for 30 minutes so today's close shows up intraday."""
    now = datetime.now(timezone.utc)
    cache = _json(SPY_CACHE, {})
    try:
        fresh = (now - datetime.fromisoformat(cache["fetched_at"])).total_seconds() < 1800
    except (KeyError, TypeError, ValueError):
        fresh = False
    if not fresh:
        try:
            import yfinance as yf
            df = yf.Ticker("SPY").history(period="2y", interval="1d")
            cache = {"date": now.date().isoformat(), "fetched_at": now.isoformat(),
                     # yfinance can return the current session with a NaN close
                     "closes": {str(i.date()): float(c) for i, c in df["Close"].items() if c == c}}
            os.makedirs(os.path.dirname(SPY_CACHE), exist_ok=True)
            with open(SPY_CACHE, "w") as f:
                json.dump(cache, f)
        except Exception:
            cache = cache or {"closes": {}}
    closes = {d: c for d, c in cache.get("closes", {}).items() if isinstance(c, (int, float)) and c == c}
    prior = [d for d in closes if d < start]
    keep = max(prior) if prior else start
    return {d: c for d, c in closes.items() if d >= keep}


# ---------------------------------------------------------------------- #
# Data
# ---------------------------------------------------------------------- #

OUTCOME_LABELS = {
    "trade_opened": ("Bought", "green"), "position_closed": ("Sold", "blue"),
    "hold_position": ("Holding", "secondary"), "low_confidence": ("Skipped · low confidence", "yellow"),
    "hold": ("Skipped · Jev said hold", "yellow"), "risk_rejected": ("Blocked by risk", "orange"),
    "jev_failed": ("Jev error", "red"), "execution_failed": ("Order failed", "red"),
    "halted": ("Halted", "red"), "market_closed": ("Market closed", "secondary"),
    "would_trade": ("Would buy (monitor)", "secondary"), "would_exit": ("Would sell (monitor)", "secondary"),
}


def _tail_csv(path: str, n: int = 3000) -> list:
    """Last n rows of a CSV without reading the whole file (the audit log
    grows by a few hundred rows per trading day)."""
    if not os.path.exists(path):
        return []
    from collections import deque
    with open(path) as f:
        header = f.readline()
        tail = deque(f, maxlen=n)
    return list(csv.DictReader([header, *tail]))


def _tz_abbr(tz: str) -> str:
    from zoneinfo import ZoneInfo
    try:
        return datetime.now(ZoneInfo(tz)).strftime("%Z")
    except Exception:
        return "UTC"

def _market_days(client):
    if client is None:
        return None
    now = datetime.now(timezone.utc)
    cache = _json(CALENDAR_CACHE, {}) or {}
    if cache.get('day') == now.date().isoformat() and isinstance(cache.get('days'), list):
        return cache['days']
    try:
        start = now.date().replace(day=1)
        end = start + timedelta(days=95)
        days = [str(item.date)[:10] for item in client.get_calendar(start=start.isoformat(), end=end.isoformat())]
        os.makedirs(os.path.dirname(CALENDAR_CACHE), exist_ok=True)
        tmp = CALENDAR_CACHE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'day': now.date().isoformat(), 'days': days}, f)
        os.replace(tmp, CALENDAR_CACHE)
        return days
    except Exception:
        return None


def decisions_today(path: Optional[str] = None, tz: str = "America/Los_Angeles") -> dict:
    """Latest decision per symbol on the most recent trading day in the audit log."""
    from zoneinfo import ZoneInfo
    rows = [r for r in _tail_csv(path or _p("journal", "audit_log.csv"))
            if r.get("cycle_outcome") and r.get("cycle_outcome") != "no_signal"]
    if not rows:
        return {"date": None, "rows": [], "cycles": 0}
    zone = ZoneInfo(tz)

    def local(ts):
        try:
            return datetime.fromisoformat(ts).astimezone(zone)
        except (TypeError, ValueError):
            return None
    day = max((t.date() for t in (local(r["timestamp"]) for r in rows) if t), default=None)
    today = [(local(r["timestamp"]), r) for r in rows]
    today = [(t, r) for t, r in today if t and t.date() == day]
    latest = {}
    for t, r in today:
        latest[r["symbol"]] = (t, r)
    out = []
    for sym, (t, r) in sorted(latest.items(), key=lambda kv: kv[1][0], reverse=True):
        label, tone = OUTCOME_LABELS.get(r["cycle_outcome"], (r["cycle_outcome"].replace("_", " "), "secondary"))
        out.append({"time": t.strftime("%H:%M"), "symbol": sym, "outcome": label, "tone": tone,
                    "jev": r.get("jev_action") or "", "jev_conf": _f(r.get("jev_confidence")),
                    "advisor": r.get("advisor_recommendation") or "", "advisor_conf": _f(r.get("advisor_confidence")),
                    "advisor_error": r.get("advisor_error") or "",
                    "first": min(tt for tt, rr in today if rr["symbol"] == sym).strftime("%H:%M"),
                    "n": sum(1 for _, rr in today if rr["symbol"] == sym)})
    return {"date": day.isoformat(), "rows": out,
            "cycles": len({t.strftime("%H:%M") for t, _ in today}), "tz": t.tzname()}


def collect(config: dict, account: Optional[dict] = None, positions: Optional[list] = None,
            fetch_benchmark: bool = True, trading_client=None) -> dict:
    from engine import strategy_from_config
    eq_rows = _csv(_p("journal", "equity_curve.csv"))
    curve = [(r["date"], _f(r.get("equity"))) for r in eq_rows if _f(r.get("equity"))]
    bench = spy_closes(curve[0][0]) if (curve and fetch_benchmark) else {}

    strategy, label = strategy_from_config(config)
    etf_on = config.get("strategy", {}).get("mode") == "etf_trend"
    name = strategy.name
    if etf_on:
        from strategies.etf_trend import VERSION, params_from_config
        ep = params_from_config(config)
        label = VERSION
        name = (f"{len(ep['symbols'])} ETFs · {ep['lookback_months']}-month trend ±{ep['band'] * 100:g}% · "
                f"{'inverse-vol' if ep['weighting'] == 'invvol' else 'equal'} weights · {len(ep['tranche_days'])} tranches")
    pos_state = _json(_p("journal", "positions_state.json"), {}) or {}
    pos = []
    for p in positions or []:
        sym = p.get("symbol")
        st = pos_state.get(sym, {})
        pos.append({"symbol": sym, "qty": _f(p.get("qty"), 0), "entry": _f(p.get("avg_entry_price")),
                    "price": _f(p.get("current_price")), "pl": _f(p.get("unrealized_pl")),
                    "pl_pct": (_f(p.get("unrealized_plpc"), 0) or 0) * 100,
                    "stop": _f(st.get("stop")), "strategy": st.get("strategy_version", "?"),
                    "mv": _f(p.get("market_value"))})
    if positions is None:  # no broker view: show what the bot believes it holds
        for sym, st in pos_state.items():
            pos.append({"symbol": sym, "qty": None, "entry": _f(st.get("entry_price")), "price": None,
                        "pl": None, "pl_pct": None, "stop": _f(st.get("stop")),
                        "strategy": st.get("strategy_version", "?")})

    closed = _csv(_p("journal", "closed_trades.csv"))
    pnl = [_f(r.get("pnl_pct")) for r in closed if _f(r.get("pnl_pct")) is not None]
    trades = [r for r in _csv(_p("journal", "trade_log.csv")) if r.get("status") != "hold"]
    if etf_on:
        trades = [r for r in trades if r.get("source") == "etf_trend"]

    cands = _csv(_p("learning", "candidates.csv"))
    lead = []
    for c in cands:
        if c.get("status") not in ("paper_trading", "promoted", "evaluating", "proposed"):
            continue
        notes = {}
        try:
            notes = json.loads(c.get("notes") or "{}")
        except ValueError:
            pass
        lead.append({"id": c["candidate_id"], "status": c["status"],
                     "offline_sharpe": notes.get("offline_sharpe"), "dsr": notes.get("offline_dsr"),
                     "paper_start": (notes.get("paper_start") or "")[:10],
                     "hypothesis": c.get("hypothesis", "")[:110]})
    reports_dir = _p("learning", "reports")
    latest_report = ""
    if os.path.isdir(reports_dir):
        files = sorted(os.listdir(reports_dir))
        if files:
            with open(os.path.join(reports_dir, files[-1])) as f:
                latest_report = f.read()
    switches = [e for e in _csv(_p("learning", "learning_audit_log.csv"))
                if e.get("event_type") in ("promoted", "promotion_announced", "promotion_vetoed", "rolled_back")]

    risk = _json(_p("journal", "risk_state.json"), {}) or {}
    exposure = (sum(_f(p.get("market_value"), 0) or 0 for p in positions) if positions is not None
                else (_f(eq_rows[-1].get("exposure")) if eq_rows else None))
    equity = _f((account or {}).get("equity")) or (curve[-1][1] if curve else None)
    last_eq = _f((account or {}).get("last_equity"))
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "mode": "paper" if os.getenv("APCA_PAPER", "true").lower() == "true" else "LIVE",
        "auto_trade": config.get("execution", {}).get("auto_trade", False),
        "strategy": {"label": label, "name": name},
        "equity": equity, "day_pl": (equity - last_eq) if (equity and last_eq) else None,
        "start_equity": curve[0][1] if curve else None,
        "peak": _f(risk.get("peak_balance")), "halted": risk.get("halted", False),
        "halt_reason": risk.get("halt_reason"),
        "curve": curve, "bench": bench,
        "positions": pos,
        "trades": trades[-25:][::-1],
        "closed": closed[-25:][::-1],
        "closed_stats": {"n": len(pnl), "win_rate": (sum(1 for x in pnl if x > 0) / len(pnl) * 100) if pnl else None,
                         "avg": (sum(pnl) / len(pnl)) if pnl else None},
        "pending": _json(_p("learning", "pending_promotion.json")),
        "leaderboard": sorted(lead, key=lambda r: -(r["offline_sharpe"] or -99))[:15],
        "switches": switches[-10:][::-1],
        "policy": _json(_p("learning", "decision_policy.json"), {}) or {},
        "readiness": _json(_p("learning", "readiness_state.json"), {}) or {},
        "latest_report": latest_report,
        "readiness_targets": config.get("live_readiness", {}),
        "tracking_error_target": config.get("live_readiness", {}).get("etf_trend", {}).get("max_tracking_error_pct", 3.0),
        "exposure": exposure,
        "tz": config.get("schedule", {}).get("timezone", "America/Los_Angeles"),
        "tz_abbr": _tz_abbr(config.get("schedule", {}).get("timezone", "America/Los_Angeles")),
        "decisions": decisions_today(tz=config.get("schedule", {}).get("timezone", "America/Los_Angeles")),
        "etf_mode": etf_on,
        "etf": (_json(_p("journal", "etf_state.json"), {}) or {}) if etf_on else {},
        "cash": _f((account or {}).get("cash")) if etf_on else None,
        "tranche_days": config.get("etf_trend", {}).get("tranche_days", [0, 5, 10, 15]) if etf_on else None,
        "market_days": _market_days(trading_client) if etf_on else None,
        "etf_compare": _json(_p("learning", "etf_compare.json")) if etf_on else None,
        "etf_fills": _csv(_p("journal", "etf_fills.csv")) if etf_on else [],
    }


# ---------------------------------------------------------------------- #
# HTML
# ---------------------------------------------------------------------- #

def _money(x, sign=False):
    if x is None:
        return "—"
    return f"{'+' if sign and x >= 0 else ''}{'-' if x < 0 else ''}${abs(x):,.0f}"


def _pct(x, sign=True):
    return "—" if x is None else f"{x:+.2f}%" if sign else f"{x:.2f}%"


TABLER_CSS = ("https://cdn.jsdelivr.net/npm/@tabler/core@1.6.1/dist/css/tabler.min.css",
              "sha384-tT2UAGE9hxG/p5d0iGIvZ/s8El3nWWG3tfG02i8iOY5Pbf8cZZRVmcrrVs+JG5Vw")
CHARTS_JS = ("https://cdn.jsdelivr.net/npm/lightweight-charts@5.2.1/dist/lightweight-charts.standalone.production.js",
             "sha384-KkYqTZlM13Zya6fVUF3IGCBOQ1ehFeJVVgxaVvD/0MebineF9EI8Jkd3NnWNChqN")


def _table(headers, rows, empty="Nothing yet.", num_cols=(), table_class=""):
    if not rows:
        return f'<div class="card-body text-secondary">{html.escape(empty)}</div>'
    th = "".join(f'<th class="{"text-end" if i in num_cols else ""}">{html.escape(h)}</th>'
                 for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"text-end num" if i in num_cols else ""}">{c}</td>'
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return (f'<div class="table-responsive"><table class="table table-vcenter card-table {table_class}">'
            f'<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>')


def _status(ok: bool, yes="pass", no="not yet"):
    # status is never color alone: icon + word
    return (f'<span class="status-good">✓ {yes}</span>' if ok
            else f'<span class="text-secondary">○ {no}</span>')


def _local_time(ts: str, tz: Optional[str]) -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.fromisoformat(ts).astimezone(ZoneInfo(tz or "America/Los_Angeles")).strftime("%b %d %H:%M")
    except (TypeError, ValueError):
        return ts[:16].replace("T", " ")


def _reason(raw: str) -> str:
    """'signal=1 jev=buy conf=0.67 risk=allow' → 'Jev buy 0.67 · risk ok'."""
    kv = dict(p.split("=", 1) for p in raw.split() if "=" in p)
    if "jev" in kv:
        return f"Jev {kv['jev']} {kv.get('conf', '')}".strip() + (" · risk ok" if kv.get("risk") == "allow" else "")
    if "exit" in kv:
        return "exit: " + kv["exit"].replace("_", " ")
    return raw[:70]


def _badge(text, tone="secondary"):
    return f'<span class="badge bg-{tone}-lt">{html.escape(text)}</span>'


def _signed(x, text):
    """Green ▲ / red ▼ with the sign kept in the text, so it reads without color."""
    if x is None:
        return text
    if abs(x) < 1e-9:
        return f'<span class="text-secondary">{text}</span>'
    return f'<span class="{"text-green" if x > 0 else "text-red"}">{"▲" if x > 0 else "▼"} {text}</span>'


def _stat(title, value, sub):
    return (f'<div class="col-6 col-lg-3 stat"><div class="card"><div class="card-body">'
            f'<div class="subheader">{title}</div><div class="h1 mb-1 mt-1 num">{value}</div>'
            f'<div class="text-secondary">{sub}</div></div></div></div>')


def _progress(label, cur, target, ok, fmt="{:.0f}", inverse=False):
    """Readiness row: value vs target with a bar (inverse = lower is better, bar shows 'used')."""
    if cur is None or not target:
        frac, val = 0.0, "—"
    else:
        frac = max(0.0, min(1.0, cur / target))
        val = f"{fmt.format(cur)} / {fmt.format(target)}" if not inverse else f"{fmt.format(cur)} of max {fmt.format(target)}"
    tone = "red" if (inverse and frac >= 1) else ("green" if ok else "primary")
    return (f'<div class="mb-3"><div class="d-flex mb-1"><div>{html.escape(label)}</div>'
            f'<div class="ms-auto num">{val} &nbsp;{_status(ok)}</div></div>'
            f'<div class="progress progress-sm"><div class="progress-bar bg-{tone}" style="width:{frac * 100:.0f}%" '
            f'role="progressbar" aria-valuenow="{frac * 100:.0f}" aria-valuemin="0" aria-valuemax="100" '
            f'aria-label="{html.escape(label)}"></div></div></div>')


def _series(data):
    """Indexed-to-100 account and SPY series aligned on the account's dates."""
    curve = data["curve"]
    if len(curve) < 2:
        return [], []
    base = curve[0][1]
    acct = [(d, v / base * 100) for d, v in curve]
    bench = data["bench"]
    spy = []
    dates = [d for d, _ in curve]
    # As-of join: latest SPY close on or before each account date (the curve
    # can start on a weekend, and today's close may not be published yet).
    bdates = sorted(bench)
    asof, j, last = [], 0, None
    for d in dates:
        while j < len(bdates) and bdates[j] <= d:
            last = bench[bdates[j]]
            j += 1
        asof.append(last)
    b0 = next((v for v in asof if v), None)
    if b0:
        spy = [(d, v / b0 * 100) for d, v in zip(dates, asof) if v]
    return acct, spy

def _next_rebalance(data):
    etf = data.get("etf") or {}
    try:
        asof = date.fromisoformat(etf["asof"])
        last = etf.get("last_executed_decision") or ""
        days = sorted(int(k) for k in data.get("tranche_days", [0, 5, 10, 15]))
        calendar = data.get("market_days")
        approx = "" if calendar else "≈"
        if calendar:
            dates = sorted(date.fromisoformat(d) for d in calendar if d >= asof.isoformat())
            # The calendar contains all trading days, including the current month before asof.
            all_dates = sorted(date.fromisoformat(d) for d in calendar)
        else:
            first = asof.replace(day=1)
            all_dates = [first + timedelta(days=i) for i in range(90)
                         if (first + timedelta(days=i)).weekday() < 5]
        for decision in all_dates:
            if decision < asof or decision.isoformat() == last:
                continue
            month = [d for d in all_dates if d.year == decision.year and d.month == decision.month]
            if decision in month and month.index(decision) in days:
                orders = next((d for d in all_dates if d > decision), None)
                if orders:
                    fmt = lambda d: d.strftime("%a %-d %b")
                    return (f"Decision {approx}{fmt(decision)} at close → orders {approx}{fmt(orders)} morning",
                            f"last: {fmt(date.fromisoformat(last))}" if last else "last: —")
    except (KeyError, ValueError, TypeError):
        pass
    return "Awaiting ETF calendar", "last: —"

def _bot_health(data):
    etf = data.get("etf") or {}
    if data.get("halted"):
        return "✕ Drawdown halt: sells only", "Halt"
    status = str(etf.get("status") or "")
    if status.lower().startswith("blocked"):
        return "✕ Blocked: " + html.escape(status.split("—", 1)[-1].strip().removeprefix("blocked").strip(" :")), "Blocked"
    try:
        now = data.get("now") or datetime.now(timezone.utc)
        local = now.astimezone(ZoneInfo(data.get("tz") or "America/Los_Angeles"))
        ts = datetime.fromisoformat(etf["updated_at"].replace("Z", "+00:00"))
        ts = ts.astimezone(local.tzinfo)
        delta = max(0, int((local - ts).total_seconds() // 60))
        age = f"{delta} min ago" if delta < 60 else f"{delta // 60} h ago"
        if local.weekday() >= 5 or not (6 * 60 + 30 <= local.hour * 60 + local.minute <= 13 * 60):
            return f"○ Market closed · last cycle {ts.strftime('%a %H:%M')}", "Market closed"
        return ((f"✓ OK · last cycle {age}", "OK") if delta < 20 else
                (f"⚠ Stale · last cycle {age}", "Stale"))
    except (KeyError, ValueError, TypeError):
        return "⚠ Stale · last cycle unknown", "Stale"

def _etf_holdings(data):
    etf = data.get("etf") or {}
    eq = data.get("equity") or 0
    held = {p["symbol"]: p for p in data["positions"]}
    tg = etf.get("targets") or {}
    groups = [("Stocks", ("SPY", "IWM", "EFA", "EEM"), "stocks"),
              ("Bonds", ("IEF", "TLT"), "bonds"),
              ("Real assets", ("GLD", "DBC"), "real"),
              ("Real estate", ("VNQ",), "estate"),
              ("T-bills & cash", ("BIL",), "bills")]
    cash = data.get("cash")
    if cash is None and eq:
        cash = eq - sum(p.get("mv") or 0 for p in held.values())
    cash = cash or 0
    total = sum(max(0, p.get("mv") or 0) for p in held.values()) + max(0, cash)
    scale = max(eq, total, 1)
    bar = '<div class="allocation-bar" role="img" aria-label="Portfolio allocation">'
    legend = '<div class="allocation-legend">'
    for label, syms, tone in groups:
        val = sum(max(0, held.get(s, {}).get("mv") or 0) for s in syms) + (max(0, cash) if tone == "bills" else 0)
        pct = val / eq * 100 if eq else 0
        title = f"{label} {pct:.1f}% · {_money(val)}"
        if val:
            bar += f'<span class="allocation-segment {tone}" style="width:{val / scale * 100:.3f}%" title="{html.escape(title)}"></span>'
        legend += f'<span><i class="{tone}"></i>{html.escape(title)}</span>'
    bar += (f'<span class="equity-marker" style="left:{eq / scale * 100:.3f}%" title="100% of equity"></span>'
            if total > eq > 0 else '') + '</div>'
    if total > eq > 0:
        legend += f'<span>100% of equity · borrowed {(total / eq - 1) * 100:.0f}%</span>'
    legend += '</div>'
    last = etf.get("tranche_last") or {}
    recent = max(last, key=lambda k: last[k] or "", default=None)
    sig = (etf.get("signals") or {}).get(recent, {}) if recent else {}
    n_tr = etf.get("tranches") or 4
    rows = []
    syms = sorted(set(tg) | set(held) | set(etf.get("in_market") or {}),
                  key=lambda s: (s == "BIL", -(tg.get(s) or 0), s))
    tip = "The strategy checks the trend in 4 slices, on trading days 1, 6, 11 and 16 of the month; 2/4 means two slices currently hold it."
    for sym in syms:
        p = held.get(sym, {})
        mv = p.get("mv") or 0
        weight = mv / eq * 100 if eq else 0
        target = (tg.get(sym) or 0) * 100
        drift = weight - target
        n_in = (etf.get("in_market") or {}).get(sym)
        v = sig.get(sym) or {}
        diff = _f(v.get("pct_vs_avg"))
        trend = "Out" if not n_in else f"In {n_in}/{n_tr}"
        if sym not in tg and mv:
            trend += ' ' + _badge("sell next rebalance", "red")
        rows.append([f'<b>{html.escape(sym)}</b>', html.escape(ASSET_CLASS.get(sym, "not in strategy")),
                     _money(mv), f"{target:.1f}%", f"{weight:.1f}%",
                     "—" if abs(drift) < .5 else f"{drift:+.1f} pp",
                     f'<span title="{html.escape(tip)}">{trend}</span>',
                     ("—" if diff is None else _signed(diff, f"{diff:+.1f}%") +
                      (' ' + _badge("near line", "yellow") if abs(diff) <= 2 else '')),
                     _signed(p.get("pl"), f"{_money(p.get('pl'), sign=True)} · {_pct(p.get('pl_pct'))}") if p.get("pl") is not None else "—"])
    plan = etf.get("pending_plan") or []
    plan_html = ("" if not plan else '<div class="card-body border-top"><b>Planned orders</b><br>' + ", ".join(
        f'{html.escape(str(o.get("side", "")).upper())} {html.escape(str(o.get("qty", "")))} {html.escape(str(o.get("symbol", "")))}' for o in plan) + '</div>')
    return (f'<div class="col-12"><div class="card"><div class="card-header"><h3 class="card-title">Holdings</h3></div>'
            f'<div class="card-body">{bar}{legend}</div>' +
            _table(["ETF", "Asset class", "Value", "Target", "Held", "Drift", "Trend", "Price vs 10-mo avg", "Unrealized P&L"],
                   rows, "Runs after the first ETF cycle.", num_cols=(2, 3, 4, 5, 7, 8), table_class="etf-table") + plan_html + '</div></div>')

def _etf_orders(data):
    from itertools import groupby
    tz = data.get('tz') or 'America/Los_Angeles'
    fills = {r.get('order_id'): r for r in data.get('etf_fills', [])}
    trades = sorted((r for r in data.get('trades', []) if r.get('source') == 'etf_trend'),
                    key=lambda r: r.get('timestamp') or '', reverse=True)[:25]
    def day(r):
        try:
            return datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00')).astimezone(ZoneInfo(tz)).date()
        except (KeyError, ValueError, TypeError):
            return date.min
    if not trades:
        body = '<tr><td colspan="8">No ETF orders yet.</td></tr>'
    else:
        body = ''
        for rebalance, group in groupby(trades, day):
            items = list(group)
            slips = [_f(fills.get(r.get('order_id'), {}).get('slippage_bp')) for r in items]
            slips = [x for x in slips if x is not None]
            average = f' · avg slippage {sum(slips) / len(slips):.1f} bp' if slips else ''
            heading = rebalance.strftime('%a %-d %b') if rebalance != date.min else 'Unknown date'
            body += f'<tr class="table-active"><th colspan="8">{heading} · rebalance · {len(items)} orders{average}</th></tr>'
            for r in items:
                fill = fills.get(r.get('order_id'), {})
                cells = [_local_time(r.get('timestamp', ''), tz), r.get('symbol') or '',
                         r.get('action', '').upper(), r.get('qty') or '',
                         _money(_f(r.get('price'))) if _f(r.get('price')) is not None else '—',
                         f'${_f(fill.get("filled_avg_price")):,.2f}' if _f(fill.get('filled_avg_price')) is not None else '—',
                         f'{_f(fill.get("slippage_bp")):+.1f}' if _f(fill.get('slippage_bp')) is not None else '—',
                         fill.get('status') or r.get('status') or '—']
                body += '<tr>' + ''.join(f'<td>{html.escape(str(c))}</td>' for c in cells) + '</tr>'
    headers = ('Time', 'ETF', 'Side', 'Qty', 'Planned price', 'Fill price', 'Slippage (bp)', 'Status')
    return ('<div class="col-12"><div class="card"><div class="card-header"><h3 class="card-title">Recent orders</h3></div>'
            '<div class="table-responsive"><table class="table card-table etf-orders"><thead><tr>' +
            ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>' + body + '</tbody></table></div></div></div>')

def _compare_svg(series):
    all_dates = sorted({d for values in series.values() for d, _ in values})
    vals = [v for values in series.values() for _, v in values]
    if not all_dates or not vals:
        return ''
    lo, hi = _scale(vals)
    ix = {d: i for i, d in enumerate(all_dates)}
    colors = {'acct': ('var(--acct)', '2', ''), 'model': ('var(--acct)', '2', ' stroke-dasharray="6 4"'),
              'sixty_forty': ('var(--forty)', '2', ''), 'spy': ('var(--benchmark)', '1.5', '')}
    paths = []
    for key, values in series.items():
        if not values:
            continue
        color, width, dash = colors[key]
        points = ' '.join(f'{44 + ix[d] * 870 / max(1, len(all_dates) - 1):.1f},{10 + (hi - v) * 226 / (hi - lo):.1f}' for d, v in values)
        paths.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width}"{dash}/>')
    return '<svg viewBox="0 0 1000 260" width="100%" role="img" aria-label="Paper account versus model, 60/40 and SPY">' + ''.join(paths) + '</svg>'


def render_html(data: dict) -> str:
    acct, spy = _series(data)
    eq, start = data["equity"], data["start_equity"]
    total_ret = ((eq / start - 1) * 100) if (eq and start) else None
    rd, tgt = data["readiness"], data.get("readiness_targets") or {}
    pend, cs = data["pending"], data["closed_stats"]
    expo = data.get("exposure")
    dec = data.get("decisions") or {"rows": [], "cycles": 0, "date": None}

    # ---- header badges + banners --------------------------------------
    mode_badge = _badge("PAPER", "blue") if data["mode"] == "paper" else _badge("LIVE", "red")
    trade_badge = _badge("auto-trade on", "green") if data["auto_trade"] else _badge("monitor only", "secondary")
    banners = []
    if data["halted"]:
        banners.append(f'<div class="alert alert-danger"><b>✕ Trading halted</b> — '
                       f'{html.escape(data["halt_reason"] or "")}. Exits still run; new entries are blocked.</div>')
    if pend:
        banners.append(f'<div class="alert alert-warning"><b>⚑ Strategy switch scheduled</b> → '
                       f'<code>{html.escape(pend["candidate_id"])}</code> after {html.escape(pend["apply_after"][:16])} UTC '
                       f'(veto: <code>python main.py learning-veto</code>)</div>')
    if rd.get("status") == "READY_FOR_REVIEW" and (not data.get("etf_mode") or rd.get("mode") == "etf_trend"):
        banners.append('<div class="alert alert-success"><b>✓ Real-money readiness criteria met</b> — review recommended.</div>')

    # ---- stat cards ----------------------------------------------------
    since = (_signed(total_ret, _pct(total_ret)) + " since start") if total_ret is not None else "no history yet"
    today = _signed(data["day_pl"], _money(data["day_pl"], sign=True)) if data["day_pl"] is not None else "—"
    expo_txt = (f"{_money(expo)} invested · {expo / eq * 100:.1f}% of equity" if (expo and eq) else
                "nothing invested")
    stats = "".join([
        _stat("Equity", _money(eq), since),
        _stat("Today", today, "vs previous close"),
        _stat("Open positions", str(len(data["positions"])), expo_txt),
        _stat("Closed trades", str(cs["n"]),
              f"win rate {cs['win_rate']:.0f}% · avg {cs['avg']:+.2f}%" if cs["n"] else "none yet"),
    ])
    etf = data.get("etf") or {}
    if data.get("etf_mode"):
        bil = sum(p.get("mv") or 0 for p in data["positions"] if p["symbol"] == "BIL")
        invested = sum(p.get("mv") or 0 for p in data["positions"] if p["symbol"] != "BIL")
        cash = data.get("cash")
        if cash is None and eq:
            cash = eq - invested - bil
        cash = cash or 0
        borrowed = cash < 0
        allocation = (f'{invested / eq * 100:.0f}% invested · ' +
                      (f'0% T-bills · {_badge(f"⚠ borrowed {-cash / eq * 100:.0f}%", "yellow")}' if borrowed else
                       f'{bil / eq * 100:.0f}% T-bills · {cash / eq * 100:.0f}% cash')) if eq else "—"
        next_value, next_sub = _next_rebalance(data)
        health, health_word = _bot_health(data)
        stats = "".join((_stat("Equity", _money(eq), f"{since} · today {today}"),
                         _stat("Allocation", allocation, f"target {etf['gross_target'] * 100:.0f}%" if borrowed and isinstance(etf.get("gross_target"), (int, float)) else ""),
                         _stat("Next rebalance", next_value, next_sub),
                         _stat("Bot health", health, "")))

    # ---- tables --------------------------------------------------------
    pos_rows = []
    for p in data["positions"]:
        room = ((p["price"] - p["stop"]) / p["price"] * 100) if (p["price"] and p["stop"]) else None
        pos_rows.append([
            f'<b>{html.escape(p["symbol"])}</b>',
            "—" if p["qty"] is None else f"{p['qty']:g}",
            "—" if p["entry"] is None else f"{p['entry']:.2f}",
            "—" if p["price"] is None else f"{p['price']:.2f}",
            _signed(p["pl"], _money(p["pl"], sign=True)) if p["pl"] is not None else "—",
            _signed(p["pl_pct"], _pct(p["pl_pct"])) if p["pl_pct"] is not None else "—",
            "—" if p["stop"] is None else f"{p['stop']:.2f}",
            "—" if room is None else f"{room:.1f}%",
        ])

    # ---- ETF target allocation ------------------------------------------
    etf = data.get("etf") or {}
    alloc_card = ""
    if data.get("etf_mode"):
        held = {p["symbol"]: (p.get("mv") or 0) for p in data["positions"]}
        last = etf.get("tranche_last") or {}
        recent = max(last, key=lambda k: last[k] or "", default=None)
        sig = (etf.get("signals") or {}).get(recent, {}) if recent else {}
        n_tr = etf.get("tranches") or 4
        alloc_rows = []
        tg = etf.get("targets") or {}
        universe = list(dict.fromkeys(list(tg) + list(etf.get("in_market") or {})))
        for sym, w in sorted(((s_, tg.get(s_, 0.0)) for s_ in universe), key=lambda kv: -kv[1]):
            v = sig.get(sym)
            n_in = (etf.get("in_market") or {}).get(sym)
            alloc_rows.append([
                f'<b>{html.escape(sym)}</b>', f'<span class="text-secondary">{html.escape(ASSET_CLASS.get(sym, ""))}</span>',
                f"{w * 100:.1f}%", f"{held.get(sym, 0) / eq * 100:.1f}%" if eq else "—",
                "cash slot" if n_in is None else (_badge(f"{n_in}/{n_tr} in", "green" if n_in == n_tr else
                                                         "secondary" if n_in == 0 else "yellow")),
                "—" if not v else _signed(v["pct_vs_avg"], f"{v['pct_vs_avg']:+.1f}%")])
        for sym in sorted(set(held) - set(etf.get("targets") or {})):
            alloc_rows.append([f'<b>{html.escape(sym)}</b>', '<span class="text-secondary">not in strategy</span>',
                               "0.0%", f"{held[sym] / eq * 100:.1f}%" if eq else "—", _badge("sell next rebalance", "red"), "—"])
        plan = etf.get("pending_plan") or []
        plan_html = ("" if not plan else '<div class="card-body border-top"><b>Planned orders</b> · ' +
                     html.escape(etf.get("status", "")) + "<br>" + ", ".join(
                         f'{html.escape(o["side"].upper())} {o["qty"]:g} {html.escape(o["symbol"])}' for o in plan) + "</div>")
        alloc_card = (f'<div class="col-12"><div class="card"><div class="card-header"><h3 class="card-title">Target allocation '
                      f'<span class="text-secondary fw-normal">· {html.escape(etf.get("strategy_version", ""))}, data through '
                      f'{html.escape(str(etf.get("asof", "—")))}, last rebalance {html.escape(str(etf.get("last_executed_decision", "—")))}'
                      f'</span></h3></div>'
                      + _table(["ETF", "Asset class", "Target", "Held", "Trend signal", "Price vs 10-mo avg"], alloc_rows,
                               "Runs after the first ETF cycle.", num_cols=(2, 3, 5))
                      + plan_html + '</div></div>')
        alloc_card = _etf_holdings(data)

    def conf(a, c):
        return (f'<span class="nowrap">{html.escape(a)} <span class="text-secondary num">{c:.2f}</span></span>'
                if (a and c is not None) else (html.escape(a) or "—"))
    dec_rows = [[f'<span class="num">{r["time"]}</span>', f'<b>{html.escape(r["symbol"])}</b>', _badge(r["outcome"], r["tone"]),
                 conf(r["jev"], r["jev_conf"]),
                 conf(r["advisor"], r["advisor_conf"]) if r["advisor"] else
                 '<span class="text-secondary">not called</span>',
                 f'<span class="text-secondary">{r["n"]} checks since {r["first"]}</span>']
                for r in dec["rows"]]
    trade_rows = [[f'<span class="num">{html.escape(_local_time(r.get("timestamp", ""), data.get("tz")))}</span>', f"<b>{html.escape(r.get('symbol',''))}</b>",
                   _badge(r.get("action", "").upper(), "green" if r.get("action") == "buy" else "blue"),
                   html.escape(str(r.get("qty", ""))), html.escape(str(r.get("price", ""))[:8]),
                   html.escape(r.get("status", "")), f'<span class="text-secondary">{html.escape(_reason(r.get("reason", "")))}</span>']
                  for r in data["trades"]]
    closed_rows = [[html.escape(r.get("closed_at", "")[:10]), f"<b>{html.escape(r.get('symbol',''))}</b>",
                    _signed(_f(r.get("pnl_pct")), _pct(_f(r.get("pnl_pct")))),
                    _signed(_f(r.get("pnl_usd")), _money(_f(r.get("pnl_usd")), sign=True)) if r.get("pnl_usd") else "—",
                    html.escape(r.get("hold_days", "")), html.escape(r.get("exit_reason", "").replace("_", " "))]
                   for r in data["closed"]]
    lead_rows = [[f"<code>{html.escape(c['id'])}</code>", _badge(c["status"].replace("_", " "),
                  "green" if c["status"] == "paper_trading" else "secondary"),
                  "—" if c["offline_sharpe"] is None else f"{c['offline_sharpe']:.2f}",
                  "—" if c["dsr"] is None else f"{c['dsr']:.2f}", html.escape(c["paper_start"] or "—"),
                  f'<span class="text-secondary">{html.escape(c["hypothesis"])}</span>'] for c in data["leaderboard"]]
    sw_rows = [[html.escape(e.get("timestamp", "")[:16].replace("T", " ")),
                html.escape(e.get("event_type", "").replace("_", " ")),
                f"<code>{html.escape(e.get('candidate_id',''))}</code>",
                html.escape((e.get("rationale") or "")[:90])] for e in data["switches"]]

    # ---- readiness -----------------------------------------------------
    checks = rd.get("checks", {})
    labels = {"min_days": "Paper history length", "min_closed_trades": "Closed trades",
              "min_sharpe": "Sharpe ≥ target", "max_drawdown_pct": "Max drawdown within limit",
              "min_psr": "Statistical confidence (PSR)", "beats_benchmark": "Beats SPY (Sharpe)",
              "min_days_since_switch": "Strategy stable"}
    if checks and rd.get("mode") == "etf_trend" and data.get("etf_mode"):
        t2 = rd.get("targets") or {}
        fmt_dd = lambda v: "—" if v is None else f"{v:.1f}%"
        if "drawdown_vs_spy" in checks:
            benchmark, benchmark_dd, benchmark_ok = "SPY", rd.get("spy_max_drawdown_pct"), checks["drawdown_vs_spy"]
        else:
            benchmark, benchmark_dd, benchmark_ok = "60/40", rd.get("sixty_forty_max_drawdown_pct"), checks.get("drawdown_vs_60_40", False)
        dd_vs = f'{fmt_dd(rd.get("max_drawdown_pct"))} vs {fmt_dd(benchmark_dd)}'
        prog = "".join([
            _progress("Trading days on this strategy", rd.get("days"), t2.get("min_days"), checks.get("min_days", False)),
            _progress("Tracking error vs model (per year)", rd.get("tracking_error_pct"), t2.get("max_tracking_error_pct"),
                      checks.get("tracking_error", False), "{:.2f}%", inverse=True),
            _progress("Max drawdown", rd.get("max_drawdown_pct"), t2.get("max_drawdown_pct"),
                      checks.get("max_drawdown_pct", False), "{:.1f}%", inverse=True),
            f'<div class="d-flex"><div>Drawdown no deeper than {benchmark} ({dd_vs})</div>'
            f'<div class="ms-auto">{_status(benchmark_ok)}</div></div>',
        ])
    elif data.get("etf_mode"):
        prog = '<p class="text-secondary mb-0">First ETF readiness check runs after today\'s end-of-day job</p>'
    elif checks:
        prog = "".join([
            _progress("Trading days of paper history", rd.get("days"), tgt.get("min_days"), checks.get("min_days", False)),
            _progress("Closed trades", rd.get("closed_trades"), tgt.get("min_closed_trades"), checks.get("min_closed_trades", False)),
            _progress("Sharpe ratio", rd.get("sharpe"), tgt.get("min_sharpe"), checks.get("min_sharpe", False), "{:.2f}"),
            _progress("Statistical confidence (PSR)", rd.get("psr"), tgt.get("min_psr"), checks.get("min_psr", False), "{:.2f}"),
            _progress("Max drawdown", rd.get("max_drawdown_pct"), tgt.get("max_drawdown_pct"),
                      checks.get("max_drawdown_pct", False), "{:.1f}%", inverse=True),
            _progress("Days on current strategy", rd.get("days_since_strategy_switch"), tgt.get("min_days_since_switch"),
                      checks.get("min_days_since_switch", False)),
            f'<div class="d-flex"><div>Beats SPY (Sharpe {rd.get("sharpe", 0):.2f} vs {rd.get("benchmark_sharpe", 0):.2f})</div>'
            f'<div class="ms-auto">{_status(checks.get("beats_benchmark", False))}</div></div>',
        ])
    else:
        prog = '<p class="text-secondary mb-0">Runs at end of day.</p>'
    status = (rd.get("status", "NOT_READY") if not data.get("etf_mode") or rd.get("mode") == "etf_trend"
              else "NOT_READY").replace("_", " ")
    pol = data["policy"]
    if data.get("etf_mode"):
        ev = pol.get("evidence") or {}
        pol_html = (f'<dl class="row mb-0"><dt class="col-6 fw-normal text-secondary">Status</dt><dd class="col-6">'
                    f'{html.escape(str(pol.get("status", "not fitted yet")).replace("_", " "))}</dd>'
                    f'<dt class="col-6 fw-normal text-secondary">Matured reviews</dt><dd class="col-6 num">{pol.get("samples", 0)}</dd></dl>'
                    f'<p class="text-secondary mt-2 mb-0">Jev reviews new ETF entries after they are placed. '
                    + (f'Entries it agreed with averaged {ev.get("taken_mean_ret_pct")}%, ones it would have blocked '
                       f'{ev.get("vetoed_mean_ret_pct")}% over 10 days.' if ev else
                       'Scores appear once 30 reviews have matured.') + '</p>')
    else:
      pol_html = (f'<dl class="row mb-0"><dt class="col-6 fw-normal text-secondary">Status</dt><dd class="col-6">'
                f'{html.escape(str(pol.get("status", "not fitted yet")).replace("_", " "))}</dd>'
                f'<dt class="col-6 fw-normal text-secondary">Matured signals</dt><dd class="col-6 num">{pol.get("samples", 0)}</dd>'
                f'<dt class="col-6 fw-normal text-secondary">Entry confidence threshold</dt><dd class="col-6">'
                f'{pol.get("min_confidence_entry", "config default")}</dd>'
                f'<dt class="col-6 fw-normal text-secondary">Jev may veto</dt><dd class="col-6">{pol.get("jev_can_veto", True)}</dd></dl>'
                + (f'<p class="text-secondary mt-2 mb-0">Taken signals avg {pol["evidence"].get("taken_mean_ret_pct")}% · '
                   f'vetoed avg {pol["evidence"].get("vetoed_mean_ret_pct")}% over the forecast horizon</p>'
                   if pol.get("evidence") else ""))

    if data.get("etf_mode"):
        jev_title = "Jev shadow scoring <span class='text-secondary fw-normal'>· logged, never blocks orders</span>"
        switch_html = lead_html = ""
    else:
        jev_title = "Is Jev helping?"
        switch_html = ('<div class="card-header border-top"><h3 class="card-title">Strategy switches</h3></div>'
                       + _table(['Time', 'Event', 'Candidate', 'Note'], sw_rows, 'No switches yet.'))
        lead_html = ('<div class="col-12"><div class="card"><div class="card-header"><h3 class="card-title">Learning leaderboard</h3></div>'
                     + _table(['Candidate', 'Status', 'Offline Sharpe', 'Deflated Sharpe', 'Paper since', 'Hypothesis'], lead_rows,
                              'No candidates in evaluation or shadow paper right now — see the latest learning report for what '
                              'was tried and why it was rejected.', num_cols=(2, 3)) + '</div></div>')
    jev_card = (f'<div class="col-lg-6"><div class="card"><div class="card-body"><details><summary>{jev_title}</summary>'
                f'{pol_html}</details></div></div></div>' if data.get("etf_mode") else
                f'<div class="col-lg-6"><div class="card"><div class="card-header"><h3 class="card-title">{jev_title}</h3></div>'
                f'<div class="card-body">{pol_html}</div>{switch_html}</div></div>')

    # ---- charts (SVG fallback is replaced by Lightweight Charts when it loads) ----
    has_chart = bool(acct)
    eq_fallback = _equity_svg(acct, spy) if has_chart else (
        '<p class="text-secondary mb-0">The equity curve starts after the first trading cycle on Hermes.</p>')
    dd_fallback = _drawdown_svg(acct) if has_chart else ""
    last_a = f"{acct[-1][1]:.2f}" if acct else "—"
    last_s = f"{spy[-1][1]:.2f}" if spy else "—"
    spy_key = (f'<span class="key"><i style="background:var(--spy)"></i>SPY buy &amp; hold '
               f'<b class="num" id="lv-s">{last_s}</b></span>') if spy else ""
    table_rows = [[d, f'{a:.2f}', next((f'{v:.2f}' for dd_, v in spy if dd_ == d), '—')] for d, a in acct[::-1][:60]]
    payload = json.dumps({"acct": acct, "spy": spy})
    report = html.escape(data["latest_report"] or "No learning run yet.")
    dec_title = f"Decisions today <span class='text-secondary fw-normal'>· {html.escape(dec['date'] or '')}, " \
                f"{dec['cycles']} cycles ({html.escape(dec.get('tz', 'PT'))})</span>" if dec["date"] else "Decisions today"

    etf_mode = data.get("etf_mode", False)
    pos_card = (alloc_card if etf_mode else
                '<div class="col-12"><div class="card"><div class="card-header"><h3 class="card-title">Open positions</h3></div>' +
                _table(['Symbol', 'Qty', 'Entry', 'Price', 'P&L', 'P&L %', 'Stop', 'Room to stop'],
                       pos_rows, 'No open positions.', num_cols=(1, 2, 3, 4, 5, 6, 7)) + '</div></div>')
    decisions_card = ("" if etf_mode else '<div class="col-lg-6"><div class="card"><div class="card-header"><h3 class="card-title">' +
                      dec_title + '</h3></div>' + _table(['Time', 'Symbol', 'Latest', 'Jev', 'Advisor', ''], dec_rows,
                      'No entry signals or held positions today.') + '</div></div>')
    closed_card = ("" if etf_mode else '<div class="col-12"><div class="card"><div class="card-header"><h3 class="card-title">Closed trades</h3></div>' +
                   _table(['Closed', 'Symbol', 'Return', 'P&L', 'Days', 'Exit'], closed_rows, 'No closed trades yet.', num_cols=(2, 3, 4)) + '</div></div>')
    orders_card = (_etf_orders(data) if etf_mode else
                   '<div class="col-lg-6"><div class="card"><div class="card-header"><h3 class="card-title">Recent orders</h3></div>' +
                   _table([f"Time ({data.get('tz_abbr', 'PT')})", 'Symbol', 'Side', 'Qty', 'Price', 'Status', 'Reason'],
                          trade_rows, 'No orders yet.', num_cols=(3, 4)) + '</div></div>')
    if etf_mode:
        try:
            updated = datetime.fromisoformat(data['generated_at'].replace('Z', '+00:00'))
            updated = updated.replace(tzinfo=updated.tzinfo or timezone.utc).astimezone(ZoneInfo(data.get('tz') or 'America/Los_Angeles'))
            header_time, updated_ts = updated.strftime('%H:%M %Z'), updated.timestamp()
        except (ValueError, TypeError):
            header_time, updated_ts = '—', 0
        header_line = (f'ETF trend strategy {html.escape(str(data["strategy"]["label"]))}: 9 ETFs, each held only while above its 10-month average, rest in T-bills. '
                       f'updated {header_time} · <span id="updated-ago" data-updated="{updated_ts:.0f}"></span>')
        title = f'Finance Bot · {_money(eq)} · {health_word}'
    else:
        header_line = (f'Strategy <b>{html.escape(data["strategy"]["label"])}</b> · '
                       f'{html.escape(data["strategy"]["name"])} · updated {html.escape(data["generated_at"])} UTC')
        title = 'Finance Bot'

    comparison = data.get('etf_compare') if etf_mode else None
    try:
        stamp = datetime.fromisoformat(comparison['generated_at'].replace('Z', '+00:00'))
        fresh = (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)) <= timedelta(days=3)
        fresh = fresh and bool(comparison.get('series', {}).get('paper'))
    except (KeyError, TypeError, ValueError, AttributeError):
        fresh = False
    chart_title = 'Account vs SPY'
    chart_note = ('<p class="text-secondary">Model comparison appears after the end-of-day job.</p>'
                  if etf_mode else '')
    range_buttons = ''
    if fresh:
        chart_title = 'Paper account vs what it should have done'
        sets = comparison['series']
        series_data = {key: sorted((d, float(v)) for d, v in sets.get(source, {}).items())
                       for key, source in (('acct', 'paper'), ('model', 'model'),
                                           ('sixty_forty', 'sixty_forty'), ('spy', 'spy'))}
        acct, spy = series_data['acct'], series_data['spy']
        last_a = f'{acct[-1][1]:.2f}' if acct else '—'
        eq_fallback = _compare_svg(series_data)
        dd_fallback = _drawdown_svg(acct) if acct else ''
        has_chart = bool(acct)
        payload = json.dumps({"acct": acct, "spy": spy,
                              "model": series_data['model'], "sixty_forty": series_data['sixty_forty']})
        chart_legend = [('Model', 'model', 'lv-m'), ('60/40', 'forty', 'lv-f'), ('SPY', 'benchmark', 'lv-s')]
        spy_key = ''.join(f'<span class="key"><i style="background:var(--{color})"></i>{label} '
                          f'<b class="num" id="{ident}">{values[-1][1]:.2f}</b></span>'
                          for label, color, ident in chart_legend
                          for values in [series_data[{'lv-m': 'model', 'lv-f': 'sixty_forty', 'lv-s': 'spy'}[ident]]]
                          if values)
        def metric(key):
            item = comparison.get(key) or {}
            ret, dd = _f(item.get('total_return_pct')), _f(item.get('max_drawdown_pct'))
            return f'{key.replace("sixty_forty", "60/40").title()} return {_pct(ret)} · max drawdown {_pct(dd)}'
        te = _f(comparison.get('tracking_error_pct'))
        target = _f(data.get('tracking_error_target'), 3.0)
        chart_note = (f'<div class="chart-metrics">{"✓" if te is not None and te <= target else "⚠"} '
                      f'Tracking error vs model {"—" if te is None else f"{te:.2f}%/yr"} (target ≤ {target:g}%) · '
                      + ' · '.join(metric(k) for k in ('paper', 'model', 'sixty_forty', 'spy')) + '</div>')
        range_buttons = '<div class="chart-ranges" aria-label="Chart range">' + ''.join(
            f'<button type="button" class="btn btn-sm btn-outline-secondary" data-range="{x}" '
            f'aria-pressed="{str(x == "All").lower()}">{x}</button>' for x in ('1M', '3M', 'YTD', 'All')) + '</div>'
        table_rows = [[d] + [f'{dict(series_data[k]).get(d):.2f}' if d in dict(series_data[k]) else '—'
                             for k in ('acct', 'model', 'sixty_forty', 'spy')] for d, _ in acct[::-1][:60]]
    else:
        payload = json.dumps({"acct": acct, "spy": spy, "model": [], "sixty_forty": []})
    data_headers = ['Date', 'Paper', 'Model', '60/40', 'SPY'] if fresh else ['Date', 'Account', 'SPY']

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ctext y='26' font-size='25'%3E%F0%9F%93%88%3C/text%3E%3C/svg%3E">
<script>(function(){{var t=null;try{{t=localStorage.getItem('fb-theme')}}catch(e){{}}
if(t!=='light'&&t!=='dark')t=window.matchMedia&&matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';
document.documentElement.setAttribute('data-bs-theme',t)}})();</script>
<link rel="stylesheet" href="{TABLER_CSS[0]}" integrity="{TABLER_CSS[1]}" crossorigin="anonymous">
<style>
:root{{--acct:{ACCOUNT['light']};--spy:{SPY['light']};--grid:var(--tblr-border-color);--axis:var(--tblr-border-color);
--muted:var(--tblr-secondary);--ink2:var(--tblr-secondary);--surface:var(--tblr-bg-surface);--ring:var(--tblr-border-color);--forty:#eb6834;--benchmark:#888}}
[data-bs-theme=dark]{{--acct:{ACCOUNT['dark']};--spy:{SPY['dark']};--forty:#d95926;--benchmark:#aaa}}
body{{background:var(--tblr-body-bg);color:var(--tblr-body-color);font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
.num{{font-variant-numeric:tabular-nums;white-space:nowrap}}.nowrap{{white-space:nowrap}}
.allocation-bar{{display:flex;position:relative;height:18px;gap:2px;background:color-mix(in srgb,var(--tblr-secondary) 35%,transparent)}}
.allocation-segment{{height:100%}}.equity-marker{{position:absolute;top:-4px;bottom:-4px;border-left:2px solid var(--tblr-body-color)}}
.allocation-legend{{display:flex;flex-wrap:wrap;gap:12px;margin-top:10px}}.allocation-legend i{{display:inline-block;width:10px;height:10px;margin-right:4px}}
.stocks{{background:#2a78d6}}.bonds{{background:#1baf7a}}.real{{background:#eda100}}.estate{{background:#e87ba4}}.bills{{background:color-mix(in srgb,var(--tblr-secondary) 35%,transparent)}}
[data-bs-theme=dark] .stocks{{background:#3987e5}}[data-bs-theme=dark] .bonds{{background:#199e70}}[data-bs-theme=dark] .real{{background:#c98500}}[data-bs-theme=dark] .estate{{background:#d55181}}
.key{{display:inline-flex;align-items:center;gap:6px;margin-right:16px;color:var(--tblr-secondary);font-size:.8125rem}}
.key i{{display:inline-block;width:14px;height:2px;border-radius:1px}}.key b{{color:var(--tblr-body-color);font-weight:600}}
.chart-box{{position:relative}}.chart-box svg text{{fill:var(--tblr-secondary);font-size:13px}}
.status-good{{color:var(--tblr-green)}}
pre.report{{white-space:pre-wrap;font-size:.8125rem;max-height:460px;overflow:auto;margin:0}}
.card-header{{flex-wrap:wrap;row-gap:6px}}.key:last-child{{margin-right:0}}
@media (max-width:576px){{.stat .h1{{font-size:1.35rem}}.card-header .ms-auto{{margin-left:0!important;width:100%}}}}
@media (max-width:576px){{.etf-table th:nth-child(2),.etf-table td:nth-child(2),.etf-table th:nth-child(3),.etf-table td:nth-child(3),.etf-table th:nth-child(8),.etf-table td:nth-child(8){{display:none}}.etf-orders th:nth-child(4),.etf-orders td:nth-child(4),.etf-orders th:nth-child(5),.etf-orders td:nth-child(5){{display:none}}.etf-table,.etf-orders{{font-size:.68rem;width:100%;table-layout:fixed}}.etf-table td,.etf-table th,.etf-orders td,.etf-orders th{{padding:4px 2px;white-space:normal;overflow-wrap:anywhere}}.etf-table .num{{white-space:normal}}}}
details>summary{{cursor:pointer}}
</style>
<link rel="stylesheet" href="assets/finbot-theme.css">
</head>
<body><div class="page">
<div class="page-wrapper">
<div class="page-header d-print-none"><div class="container-xl"><div class="row g-2 align-items-center">
  <div class="col"><div class="d-flex flex-wrap align-items-center gap-2"><h2 class="page-title m-0">Finance Bot</h2>{mode_badge}{trade_badge}</div>
  <div class="text-secondary mt-1">{header_line}</div></div>
  <div class="col-auto"><button type="button" class="btn btn-ghost-secondary" id="theme" aria-label="Toggle light or dark theme">◐ Theme</button></div>
</div></div></div>
<div class="page-body"><div class="container-xl">
{''.join(banners)}
<div class="row row-deck row-cards">{stats}

<div class="col-12"><div class="card">
  <div class="card-header"><h3 class="card-title">{chart_title} <span class="text-secondary fw-normal">· indexed to 100 at start</span></h3>
  <div class="ms-auto"><span class="key"><i style="background:var(--acct)"></i>Paper account <b class="num" id="lv-a">{last_a}</b></span>{spy_key}</div></div>
  <div class="card-body">
    {range_buttons}<div class="chart-box" id="eq" style="min-height:{280 if has_chart else 0}px" role="img" aria-label="Account equity versus SPY, indexed to 100">{eq_fallback}</div>
    {chart_note}
    {('<div class="subheader mt-3 mb-1">Drawdown from peak</div><div class="chart-box" id="dd" style="min-height:110px" role="img" aria-label="Drawdown from peak">' + dd_fallback + '</div>') if has_chart else ''}
    {('<details class="mt-3"><summary class="text-secondary">Data table</summary>' + _table(data_headers, table_rows, num_cols=tuple(range(1, len(data_headers)))) + '</details>') if has_chart else ''}
  </div></div></div>

{pos_card}
{decisions_card}

{orders_card}
{closed_card}

<div class="col-lg-6"><div class="card"><div class="card-header"><h3 class="card-title">Real-money readiness</h3>
  <div class="ms-auto">{_badge(status, 'green' if rd.get('status') == 'READY_FOR_REVIEW' else 'secondary')}</div></div>
  <div class="card-body">{prog}<p class="text-secondary mt-3 mb-0">The bot never switches to real money by itself.</p></div></div></div>
{jev_card}
{lead_html}

<div class="col-12"><div class="card"><div class="card-body"><details><summary><b>Latest learning report</b></summary>
<pre class="report mt-3">{report}</pre></details></div></div></div>
</div></div></div>
<footer class="footer footer-transparent d-print-none"><div class="container-xl text-secondary">
Read-only view · charts by <a href="https://www.tradingview.com/" target="_blank" rel="noopener">TradingView Lightweight Charts</a></div></footer>
</div></div>
<script src="{CHARTS_JS[0]}" integrity="{CHARTS_JS[1]}" crossorigin="anonymous"></script>
<script>
(function(){{
const D={payload};
const root=document.documentElement;
const THEME={{light:{{acct:'#1f5fbf',spy:'#c45a22',forty:'#c45a22',benchmark:'#8c877b',text:'#8c877b',grid:'#e6e1d6'}},
              dark:{{acct:'#6fa3ee',spy:'#e08a52',forty:'#e08a52',benchmark:'#77736a',text:'#77736a',grid:'#24231f'}}}};
const btn=document.getElementById('theme');
const charts=[];let series={{}};
const MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
function day(t,yr){{let y,m,d;if(typeof t==='string'){{[y,m,d]=t.split('-').map(Number)}}
  else if(typeof t==='object'){{y=t.year;m=t.month;d=t.day}}else{{const x=new Date(t*1000);y=x.getUTCFullYear();m=x.getUTCMonth()+1;d=x.getUTCDate()}}
  return MONTHS[m-1]+' '+d+(yr?', '+y:'')}}
const fit=c=>c.timeScale().setVisibleLogicalRange({{from:0,to:Math.max(1,D.acct.length-1)}});
function mode(){{return root.getAttribute('data-bs-theme')==='dark'?'dark':'light'}}
function opts(){{const t=THEME[mode()];return {{autoSize:true,
  layout:{{background:{{type:'solid',color:'transparent'}},textColor:t.text,fontFamily:'"IBM Plex Mono",ui-monospace,monospace',fontSize:11}},
  grid:{{vertLines:{{visible:false}},horzLines:{{color:t.grid}}}},
  rightPriceScale:{{borderVisible:false}},
  timeScale:{{borderVisible:false,fixLeftEdge:true,fixRightEdge:true,lockVisibleTimeRangeOnResize:true,
    tickMarkFormatter:t=>day(t,false)}},
  localization:{{timeFormatter:t=>day(t,true)}},
  crosshair:{{mode:0}},handleScroll:false,handleScale:false}}}}
function recolor(){{const t=THEME[mode()];charts.forEach((c,i)=>{{const o=opts();if(i)o.layout.attributionLogo=false;c.applyOptions(o)}});
  if(series.a)series.a.applyOptions({{color:t.acct}});if(series.s)series.s.applyOptions({{color:D.model.length?t.benchmark:t.spy}});
  if(series.m)series.m.applyOptions({{color:t.acct}});if(series.f)series.f.applyOptions({{color:t.forty}});
  if(series.d)series.d.applyOptions({{topLineColor:t.acct,bottomLineColor:t.acct,bottomFillColor1:t.acct+'40',bottomFillColor2:t.acct+'08'}});}}
btn.addEventListener('click',()=>{{const n=mode()==='dark'?'light':'dark';root.setAttribute('data-bs-theme',n);
  try{{localStorage.setItem('fb-theme',n)}}catch(e){{}}recolor();}});
const LWC=window.LightweightCharts;
if(!LWC||!D.acct.length)return;  // keep the static SVG fallback
const pt=a=>a.map(([d,v])=>({{time:d,value:v}}));
const eqBox=document.getElementById('eq'),ddBox=document.getElementById('dd');
eqBox.innerHTML='';eqBox.style.height='280px';
const c1=LWC.createChart(eqBox,opts());charts.push(c1);
const t=THEME[mode()];
const fmt={{type:'price',precision:2,minMove:0.01}};
series.a=c1.addSeries(LWC.LineSeries,{{color:t.acct,lineWidth:2,title:'Account',priceFormat:fmt,lastValueVisible:true,priceLineVisible:false}});
series.a.setData(pt(D.acct));
if(D.model.length){{series.m=c1.addSeries(LWC.LineSeries,{{color:t.acct,lineWidth:2,lineStyle:2,title:'Model',priceFormat:fmt,lastValueVisible:true,priceLineVisible:false}});
  series.m.setData(pt(D.model));}}
if(D.sixty_forty.length){{series.f=c1.addSeries(LWC.LineSeries,{{color:t.forty,lineWidth:2,title:'60/40',priceFormat:fmt,lastValueVisible:true,priceLineVisible:false}});
  series.f.setData(pt(D.sixty_forty));}}
if(D.spy.length){{series.s=c1.addSeries(LWC.LineSeries,{{color:D.model.length?t.benchmark:t.spy,lineWidth:D.model.length?1.5:2,title:'SPY',priceFormat:fmt,lastValueVisible:true,priceLineVisible:false}});
  series.s.setData(pt(D.spy));}}
fit(c1);
let peak=-1e18;const dd=D.acct.map(([d,v])=>{{peak=Math.max(peak,v);return {{time:d,value:(v/peak-1)*100}}}});
if(ddBox){{ddBox.innerHTML='';ddBox.style.height='110px';
  const c2=LWC.createChart(ddBox,{{...opts(),layout:{{...opts().layout,attributionLogo:false}}}});charts.push(c2);
  series.d=c2.addSeries(LWC.BaselineSeries,{{baseValue:{{type:'price',price:0}},lineWidth:2,
    topLineColor:t.acct,topFillColor1:'rgba(0,0,0,0)',topFillColor2:'rgba(0,0,0,0)',
    bottomLineColor:t.acct,bottomFillColor1:t.acct+'40',bottomFillColor2:t.acct+'08',
    priceFormat:{{type:'custom',formatter:v=>v.toFixed(1)+'%',minMove:0.01}},title:'Drawdown',priceLineVisible:false}});
  series.d.setData(dd);fit(c2);
  // keep both charts on the same time window
  c1.timeScale().subscribeVisibleLogicalRangeChange(r=>r&&c2.timeScale().setVisibleLogicalRange(r));}}
const la=document.getElementById('lv-a'),ls=document.getElementById('lv-s');
const lastA=la.textContent,lastS=ls?ls.textContent:'';
const lm=document.getElementById('lv-m'),lf=document.getElementById('lv-f');
const lastM=lm?lm.textContent:'',lastF=lf?lf.textContent:'';
document.querySelectorAll('[data-range]').forEach(b=>b.addEventListener('click',()=>{{
  const end=D.acct[D.acct.length-1][0],year=Number(end.slice(0,4));
  const start=new Date(end+'T00:00:00Z');
  if(b.dataset.range==='1M')start.setUTCMonth(start.getUTCMonth()-1);
  if(b.dataset.range==='3M')start.setUTCMonth(start.getUTCMonth()-3);
  if(b.dataset.range==='YTD')start.setUTCFullYear(year,0,1);
  if(b.dataset.range==='All')c1.timeScale().fitContent();
  else c1.timeScale().setVisibleRange({{from:start.toISOString().slice(0,10),to:end}});
  document.querySelectorAll('[data-range]').forEach(x=>x.setAttribute('aria-pressed',x===b?'true':'false'));
}}));
c1.subscribeCrosshairMove(p=>{{if(!p||!p.time||!p.seriesData){{la.textContent=lastA;if(ls)ls.textContent=lastS;
  if(lm)lm.textContent=lastM;if(lf)lf.textContent=lastF;return}}
  const a=p.seriesData.get(series.a),s=series.s&&p.seriesData.get(series.s);
  const m=series.m&&p.seriesData.get(series.m),f=series.f&&p.seriesData.get(series.f);
  la.textContent=a?a.value.toFixed(2):lastA;if(ls)ls.textContent=s?s.value.toFixed(2):lastS;
  if(lm)lm.textContent=m?m.value.toFixed(2):lastM;if(lf)lf.textContent=f?f.value.toFixed(2):lastF;}});
}})();
</script></body></html>"""


def _readable(summary: str, labels: dict) -> str:
    for k, v in labels.items():
        summary = summary.replace(k, v.lower())
    return summary


def _cls_num(x, text):
    if x is None:
        return text
    return f'<span class="{"pos" if x >= 0 else "neg"}">{text}</span>'


def _scale(vals, lo_pad=0.02):
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    pad = (hi - lo) * lo_pad
    return lo - pad, hi + pad


def _equity_svg(acct, spy, W=1000, H=260, L=44, R=86, T=10, B=24):
    n = len(acct)
    allv = [v for _, v in acct] + [v for _, v in spy]
    lo, hi = _scale(allv)
    x = lambda i: L + i * (W - L - R) / max(1, n - 1)
    y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    idx = {d: i for i, (d, _) in enumerate(acct)}
    grid = ""
    dec = 0 if hi - lo >= 4 else (1 if hi - lo >= 0.4 else 2)  # else every tick reads "100"
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        grid += (f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--grid)" stroke-width="1"/>'
                 f'<text x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.{dec}f}</text>')
    for i in sorted({0, n // 2, n - 1}):
        grid += f'<text x="{x(i):.1f}" y="{H - 6}" text-anchor="middle">{acct[i][0]}</text>'
    pa = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, (_, v) in enumerate(acct))
    ps = " ".join(f"{x(idx[d]):.1f},{y(v):.1f}" for d, v in spy if d in idx)
    labels = f'<text x="{W - R + 6}" y="{y(acct[-1][1]) + 4:.1f}" style="fill:var(--ink2)">Account {acct[-1][1]:.1f}</text>'
    if spy:
        dy = 14 if abs(y(spy[-1][1]) - y(acct[-1][1])) < 14 else 0
        sy = y(spy[-1][1]) + (dy if spy[-1][1] < acct[-1][1] else -dy)
        labels += f'<text x="{W - R + 6}" y="{sy + 4:.1f}" style="fill:var(--ink2)">SPY {spy[-1][1]:.1f}</text>'
    base = f'<line x1="{L}" x2="{W - R}" y1="{y(100):.1f}" y2="{y(100):.1f}" stroke="var(--axis)" stroke-width="1"/>' if lo < 100 < hi else ""
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" data-w="{W}" data-l="{L}" data-r="{R}" role="img" '
            f'aria-label="Account equity versus SPY, indexed to 100">{grid}{base}'
            + (f'<polyline points="{ps}" fill="none" stroke="var(--spy)" stroke-width="2" stroke-linejoin="round"/>' if ps else "")
            + f'<polyline points="{pa}" fill="none" stroke="var(--acct)" stroke-width="2" stroke-linejoin="round"/>'
            f'{labels}<line id="xh" y1="{T}" y2="{H - B}" stroke="var(--axis)" stroke-width="1" style="display:none"/></svg>')


def _drawdown_svg(acct, W=1000, H=110, L=44, R=86, T=6, B=18):
    peak, dd = -1e18, []
    for _, v in acct:
        peak = max(peak, v)
        dd.append((v / peak - 1) * 100)
    n, lo = len(dd), min(min(dd), -1.0)
    x = lambda i: L + i * (W - L - R) / max(1, n - 1)
    y = lambda v: T + (0 - v) / (0 - lo) * (H - T - B)
    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(dd))
    area = f"{L},{y(0):.1f} {pts} {x(n - 1):.1f},{y(0):.1f}"
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Drawdown from peak">'
            f'<line x1="{L}" x2="{W - R}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="var(--axis)"/>'
            f'<text x="{L - 6}" y="{y(0) + 4:.1f}" text-anchor="end">0%</text>'
            f'<text x="{L - 6}" y="{y(lo) + 4:.1f}" text-anchor="end">{lo:.1f}%</text>'
            f'<polygon points="{area}" fill="var(--acct)" fill-opacity=".18"/>'
            f'<polyline points="{pts}" fill="none" stroke="var(--acct)" stroke-width="2"/>'
            f'<text x="{W - R + 6}" y="{y(dd[-1]) + 4:.1f}" style="fill:var(--ink2)">{dd[-1]:.1f}%</text></svg>')


def render_png(data: dict, path: str = EQUITY_PNG_PATH) -> Optional[str]:
    """Account vs SPY + drawdown as a PNG for Discord (needs matplotlib)."""
    acct, spy = _series(data)
    if len(acct) < 2:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        return None
    from datetime import date
    d = [date.fromisoformat(a) for a, _ in acct]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4.6), dpi=130, sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    for a in (ax, ax2):
        a.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color("#c3c2b7")
        a.tick_params(colors="#898781", labelsize=8)
        a.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.plot(d, [v for _, v in acct], color=ACCOUNT["light"], lw=2, label="Paper account")
    if spy:
        ax.plot([date.fromisoformat(a) for a, _ in spy], [v for _, v in spy], color=SPY["light"], lw=2,
                label="SPY buy & hold")
    ax.axhline(100, color="#c3c2b7", lw=0.8)
    ax.set_title("Account vs SPY (indexed to 100)", loc="left", fontsize=10, color="#0b0b0b")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor="#52514e")
    peak, dd = -1e18, []
    for _, v in acct:
        peak = max(peak, v)
        dd.append((v / peak - 1) * 100)
    ax2.fill_between(d, dd, 0, color=ACCOUNT["light"], alpha=0.18, lw=0)
    ax2.plot(d, dd, color=ACCOUNT["light"], lw=1.5)
    ax2.set_ylabel("Drawdown %", fontsize=8, color="#52514e")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_dashboard(config: dict, account: Optional[dict] = None, positions: Optional[list] = None,
                    png: bool = False) -> str:
    data = collect(config, account, positions)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    tmp = DASHBOARD_PATH + ".tmp"
    with open(tmp, "w") as f:
        f.write(render_html(data))
    os.replace(tmp, DASHBOARD_PATH)
    if png:
        render_png(data)
    return DASHBOARD_PATH
