"""
ETF trend execution (strategy.mode = "etf_trend").

Each scheduled cycle (every 5 min, 05:45-13:00 PT):
  1. Daily closes for the ETFs up to the LAST COMPLETED session (today's
     unfinished bar is dropped), from yfinance with an Alpaca fallback.
  2. Target weights from strategies/etf_trend.targets_asof (pure function).
  3. A rebalance is due when the last completed session was a tranche day
     (trading day 1, 6, 11 or 16 of the month) and has not been executed yet,
     or on the very first run (moving the account from stocks to ETFs).
  4. Before the open the plan is shown and saved; after the open it is
     executed ONCE: sells first, then buys sized to cash. Next-day-open fills,
     like the backtest.
  5. Jev runs in shadow mode: its opinion on new entries is logged as a
     forecast and in the audit log, and never blocks an order.

decide() is shared by the live cycle and the replay, so both compute the same
orders from the same inputs.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from strategies import etf_trend as et
from strategies import overlay as ov
import ai_pm
from engine import (JOURNAL_DIR, EQUITY_CURVE_PATH, CycleLocked, cycle_lock, _read_json,
                    _write_json, _notify, _record_equity)
from journal.trade_log import log_trade, ensure_header
from journal.audit_log import log_audit_event, ensure_audit_header

logger = logging.getLogger("finance-bot.etf")
STATE_PATH = os.path.join(JOURNAL_DIR, "etf_state.json")
CACHE_DIR = os.path.join(JOURNAL_DIR, "cache")
EASTERN = ZoneInfo("America/New_York")
HISTORY_PERIOD = "2y"   # ~500 sessions: 10 monthly samples + vol window, plus band memory


# ---------------------------------------------------------------------- #
# Data
# ---------------------------------------------------------------------- #

def _naive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(EASTERN).tz_localize(None)
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")]

def _fill_interior_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for symbol in result.columns:
        original = result[symbol]
        interior = original.notna().cummax() & original.notna()[::-1].cummax()[::-1]
        gaps = original.isna() & interior
        runs = (gaps != gaps.shift(fill_value=False)).cumsum()
        short = gaps & gaps.groupby(runs).transform("sum").le(2)
        filled = original.ffill().where(short, original)
        for date in result.index[short]:
            logger.warning("Filled interior price gap: %s %s", symbol, date.date())
        result[symbol] = filled
    return result

def missing_price_data(closes: pd.DataFrame, universe: list, rows: int = 260) -> list:
    """Symbols missing a complete, current price window."""
    missing = []
    for symbol in universe:
        if (symbol not in closes or len(closes) < rows
                or closes[symbol].iloc[-rows:].isna().any()
                or closes[symbol].last_valid_index() != closes.index[-1]):
            missing.append(symbol)
    return missing


def load_prices(dp, symbols: list, now: Optional[datetime] = None, use_cache: bool = True) -> tuple:
    """(closes, opens) of completed sessions only. yfinance (adjusted,
    consolidated tape) first; Alpaca daily bars as a fallback.

    Completed sessions do not change during the day, so the result is cached
    per (day, before/after the close, symbols): one download per day instead
    of one per 5-minute cycle."""
    import hashlib
    now_et = (now or datetime.now(timezone.utc)).astimezone(EASTERN)
    after_close = now_et.time() >= datetime.strptime("16:15", "%H:%M").time()
    key = hashlib.sha1(",".join(sorted(symbols)).encode()).hexdigest()[:8]
    cache = os.path.join(CACHE_DIR, f"etf_prices_{now_et.date()}_{'pm' if after_close else 'am'}_{key}.pkl")
    if use_cache and os.path.exists(cache):
        try:
            c, o = pd.read_pickle(cache)
            filled_c, filled_o = _fill_interior_gaps(c), _fill_interior_gaps(o)
            if not c.equals(filled_c) or not o.equals(filled_o):
                pd.to_pickle((filled_c, filled_o), cache)
            c, o = filled_c, filled_o
            return c, o
        except Exception:
            pass
    closes, opens = {}, {}
    for s in symbols:
        df = pd.DataFrame()
        try:
            df = dp.fetch_yf_history(s, period=HISTORY_PERIOD, interval="1d")
        except Exception as e:  # pragma: no cover - network
            logger.warning("yfinance %s failed: %s", s, e)
        if (df is None or df.empty) and getattr(dp, "data_client", None):
            try:
                end = datetime.now(timezone.utc)
                got = dp.fetch_alpaca_bars_multi([s], end - pd.Timedelta(days=760), end)
                df = got.get(s, pd.DataFrame())
            except Exception as e:  # pragma: no cover - network
                logger.warning("Alpaca bars %s failed: %s", s, e)
        if df is None or df.empty or "Close" not in df:
            continue
        df = _naive(df)
        closes[s] = df["Close"]
        if "Open" in df:
            opens[s] = df["Open"]
    c = pd.DataFrame(closes).sort_index()
    o = pd.DataFrame(opens).reindex(c.index)
    today = pd.Timestamp(now_et.date())
    if not after_close:
        c, o = c[c.index < today], o[o.index < today]       # today's bar is not final yet
    # a session row with a missing close (data glitch) is not usable yet
    while len(c) and c.iloc[-1][[s for s in c.columns if c[s].notna().sum() > 1]].isna().any():
        c, o = c.iloc[:-1], o.iloc[:-1]
    c, o = _fill_interior_gaps(c), _fill_interior_gaps(o)
    if use_cache and len(c) and set(symbols) <= set(c.columns):   # only cache complete downloads
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            for f in os.listdir(CACHE_DIR):                         # keep the cache small
                if f.startswith("etf_prices_") and str(now_et.date()) not in f:
                    os.remove(os.path.join(CACHE_DIR, f))
            pd.to_pickle((c, o), cache)
        except OSError as e:
            logger.warning("price cache write failed: %s", e)
    return c, o


# ---------------------------------------------------------------------- #
# Decision (shared by live and replay)
# ---------------------------------------------------------------------- #

def max_exposure(cfg: dict, max_gross: Optional[float] = None) -> float:
    """Total long exposure the targets may reach (1.0 without the overlay)."""
    ovs = ov.settings(cfg)
    return (max_gross or ovs["max_gross"]) if ov.active(ovs) else min(1.0, max_gross or 1.0)


def decide(closes: pd.DataFrame, holdings: dict, prices: dict, equity: float, cash: float,
           state: dict, cfg: dict, params: dict, peak_equity: Optional[float] = None,
           max_gross: Optional[float] = None, ai: Optional[dict] = None, ai_due: bool = False) -> dict:
    """
    Pure decision step. Returns dict(due, reason, targets, info, orders, halted).
    `due` is True when the last completed session is a tranche day not yet
    executed, or the account has never been allocated (first run).
    The index/leverage overlay (config etf_trend.overlay) is applied to the
    trend targets; max_gross overrides its exposure cap (1.0 = no borrowing).
    """
    trend_targets, info = et.targets_asof(closes, None, params)
    ovs = ov.settings(cfg)
    cash_sym = params["cash_symbol"]
    targets = ov.apply(trend_targets, ovs, cash_sym, max_gross=max_gross)
    rules_targets = targets
    asof = info["asof"]
    first = not state.get("initialized")
    due = first or (info["rebalance_today"] and state.get("last_executed_decision") != asof)
    reason = "first allocation" if first else ("rebalance day" if due else "no rebalance due")
    if ai is not None and ai_due and not due:
        due, reason = True, "AI plan change"     # the AI plan changed since the last trade
    halt_pct = float(cfg.get("max_drawdown_halt_pct", 20.0))
    halted = bool(peak_equity and equity < peak_equity * (1 - halt_pct / 100))
    if ai is not None and not halted:            # AI tilts (ai_pm.py); ignored during a drawdown halt
        current_w = {s: float(q) * float(prices.get(s) or 0) / equity for s, q in holdings.items()} if equity else {}
        targets = ai_pm.apply(rules_targets, ai, current_w, max_exposure(cfg, max_gross), cash_sym)
    if halted and ov.active(ovs):
        # stop borrowing: aim for the same mix without leverage; only sells run
        targets = ov.apply(trend_targets, ovs, cash_sym, leverage=1.0, max_gross=1.0)
    orders = []
    if due:
        orders = et.plan_orders(targets, holdings, prices, equity, cash,
                                float(cfg.get("min_trade_pct", 0.5)), float(cfg.get("cash_buffer_pct", 1.0)),
                                borrow_limit=0.0 if halted else ov.borrow_limit(targets, cash_sym))
        if halted:
            orders = [o for o in orders if o["side"] == "sell"]
            reason += f" — DRAWDOWN HALT (> {halt_pct:.0f}% below peak): sells only"
    return {"due": due, "reason": reason, "targets": targets, "trend_targets": trend_targets,
            "info": info, "orders": orders, "halted": halted, "asof": asof,
            "gross": ov.gross(targets, cash_sym), "label": ov.label(cfg, et.VERSION) + ("+ai" if ai is not None else ""),
            "rules_targets": rules_targets}


# ---------------------------------------------------------------------- #
# Live cycle
# ---------------------------------------------------------------------- #

def run_etf_cycle(dp, config: dict, execute: bool, now: Optional[datetime] = None) -> dict:
    try:
        with cycle_lock():
            return _run(dp, config, execute, now)
    except CycleLocked:
        print("[SKIPPED] Previous cycle still running (lock held)")
        return {"skipped": "locked"}


def _peak_equity(current: float) -> float:
    import csv
    peak = current
    if os.path.exists(EQUITY_CURVE_PATH):
        with open(EQUITY_CURVE_PATH) as f:
            for r in csv.DictReader(f):
                try:
                    peak = max(peak, float(r.get("equity") or 0))
                except ValueError:
                    pass
    return peak


class _Exposure:
    def __init__(self, v):
        self.v = v

    def total_open_exposure(self):
        return self.v


def _run(dp, config: dict, execute: bool, now: Optional[datetime]) -> dict:
    params = et.params_from_config(config)
    cfg = config.get("etf_trend", {}) or {}
    universe = list(params["symbols"]) + [params["cash_symbol"]]
    if execute and not getattr(dp, "trading_client", None):
        print("ERROR: Alpaca trading client not configured. Cannot trade.")
        return {"error": "no_trading_client"}
    if execute and not dp.paper:
        print("ERROR: refusing to execute — account is not paper (APCA_PAPER!=true).")
        return {"error": "not_paper"}
    ensure_header()
    ensure_audit_header()

    account, positions, open_orders, market_open = {}, [], [], False
    if getattr(dp, "trading_client", None):
        try:
            account = dp.get_account()
            positions = dp.get_positions()
            open_orders = dp.get_open_orders()
            market_open = bool(dp.get_clock().get("is_open"))
        except Exception as e:
            logger.error("Broker state fetch failed: %s", e)
            if execute:
                print(f"ERROR: cannot read broker state ({e}); not trading this cycle.")
                return {"error": "broker_state"}

    closes, opens = load_prices(dp, universe, now)
    if closes.empty or len(closes) < 260:
        print("ERROR: not enough ETF price history; not trading this cycle.")
        return {"error": "no_data"}
    missing = missing_price_data(closes, universe)
    if missing:
        names = ", ".join(missing)
        print(f"ERROR: incomplete price data for {names}; not trading this cycle.")
        state = _read_json(STATE_PATH, {})
        today = (now or datetime.now(timezone.utc)).astimezone(EASTERN).date().isoformat()
        if state.get("incomplete_data_alerted_on") != today:
            _notify(config, f"🚨 Trading paused: missing price data for {names}. No orders sent.", "alerts")
            state["incomplete_data_alerted_on"] = today
        state["status"] = f"blocked — incomplete price data ({names})"
        _write_json(STATE_PATH, state)
        return {"error": "incomplete_data", "symbols": missing}

    holdings = {p["symbol"]: float(p.get("qty") or 0) for p in positions if float(p.get("qty") or 0) != 0}
    prices = {s: float(closes[s].dropna().iloc[-1]) for s in closes.columns if closes[s].notna().any()}
    for p in positions:
        try:
            prices[p["symbol"]] = float(p.get("current_price") or prices.get(p["symbol"]) or 0)
        except (TypeError, ValueError):
            pass
    equity = float(account.get("equity") or account.get("portfolio_value") or 0) if account else 0.0
    cash = float(account.get("cash") or 0) if account else 0.0
    state = _read_json(STATE_PATH, {})
    peak = _peak_equity(equity) if equity else None

    ovs = ov.settings(cfg)
    max_gross, margin_note = None, ""
    if ov.active(ovs) and ovs["leverage"] > 1 and ovs["require_margin"] and account:
        try:
            mult = float(account.get("multiplier") or 1)
        except (TypeError, ValueError):
            mult = 1.0
        if mult < 2:
            max_gross = 1.0
            margin_note = f"margin not available (account multiplier {mult:g}) — leverage capped at 1.0x"
            print(f"  WARNING: {margin_note}")
            today = datetime.now(timezone.utc).date().isoformat()
            if execute and state.get("margin_alerted") != today:
                state["margin_alerted"] = today
                _notify(config, f"⚠️ Margin is unavailable. Trading stays within available cash. "
                                f"Check the paper account before the next rebalance.", "alerts")

    d = decide(closes, holdings, prices, equity, cash, state, cfg, params, peak, max_gross=max_gross)
    ai_plan = None
    if ai_pm.active(config):
        # The AI adjusts the rule targets once a day; any failure falls back to the rules.
        try:
            current_w = ({s: q * float(prices.get(s) or 0) / equity for s, q in holdings.items()}
                         if equity else {})
            ai_plan = ai_pm.daily_plan(config, asof=d["asof"], rules=d["rules_targets"], info=d["info"],
                                       closes=closes, current_w=current_w, equity=equity, cash=cash, peak=peak,
                                       cap=max_exposure(cfg, max_gross), symbols=list(params["symbols"]),
                                       execute=execute, now=now)
        except Exception as e:
            logger.error("AI plan failed, using the rules: %s", e)
            ai_plan = {"tilts": {}, "hold": [], "status": "error", "note": type(e).__name__}
        ai_due = ai_pm.changed(ai_plan, state.get("ai_executed"), ai_pm.settings(config)["min_change"])
        d = decide(closes, holdings, prices, equity, cash, state, cfg, params, peak, max_gross=max_gross,
                   ai=ai_plan, ai_due=ai_due)
    mode = "TRADE (paper)" if execute else "MONITOR (no orders)"
    print(f"=== ETF trend cycle [{mode}] {datetime.now().isoformat()} ===")
    print(f"  Strategy: {d['label']} · data through {d['asof']} · market_open={market_open}")
    if ov.active(ovs):
        print(f"  Overlay: {ovs['index_weight'] * 100:.0f}% {ovs['index_symbol']} + "
              f"{(1 - ovs['index_weight']) * 100:.0f}% trend, x{ovs['leverage']:g} → target exposure "
              f"{d['gross'] * 100:.0f}% of equity (cap {(max_gross or ovs['max_gross']) * 100:.0f}%)")
    print(f"  Equity ${equity:,.2f} · cash ${cash:,.2f} · {len(holdings)} positions · {d['reason']}")
    tgt_txt = ", ".join(f"{s} {w * 100:.1f}%" for s, w in d["targets"].sort_values(ascending=False).items())
    print(f"  Targets: {tgt_txt}")

    if ai_plan is not None:
        print(f"  AI plan ({ai_plan['status']}{': ' + ai_plan['note'] if ai_plan.get('note') else ''}): "
              f"{ai_pm._fmt(ai_plan)}")
    executed, failures, status = [], [], "idle"
    # total long exposure the orders may reach (BIL included); 1.0 without the overlay
    exposure_cap = (max_gross or ovs["max_gross"]) if ov.active(ovs) else 1.0
    if d["due"] and d["orders"]:
        # Working entry/exit orders block a new plan (they would be duplicated).
        # Protective stops do not: the old stock bot leaves a GTC stop on every
        # position, and close_position() cancels it when that stock is sold.
        pending_universe = [o for o in open_orders if o.get("symbol") in set(universe) | set(holdings)
                            and o.get("type") not in ("stop", "stop_limit", "trailing_stop")]
        for o in d["orders"]:
            print(f"    plan: {o['side'].upper():4s} {o['qty']:g} {o['symbol']} (~${o['qty'] * o['price']:,.0f}, {o['reason']})")
        if not execute:
            status = "planned (monitor mode)"
        elif not market_open:
            status = "planned — executes after the open"
        elif pending_universe:
            status = f"waiting — {len(pending_universe)} open order(s) still working"
        elif equity and ov.projected_long(holdings, prices, d["orders"]) / equity > exposure_cap + 0.02:
            proj = ov.projected_long(holdings, prices, d["orders"]) / equity
            status = f"blocked — projected exposure {proj * 100:.0f}% is above the cap"
            print(f"  ERROR: {status}; not trading this cycle.")
            _notify(config, f"🚨 Rebalance paused: planned holdings would be {proj * 100:.0f}% of account value, "
                            f"above the {exposure_cap * 100:.0f}% safety limit. No orders sent.", "alerts")
        else:
            executed, failures = _execute(dp, config, d, universe, closes)
            status = "executed" if not failures else f"executed with {len(failures)} failure(s)"
    elif d["due"]:
        status = "nothing to trade (already at target)"
        if execute and market_open:
            executed = []

    if d["due"] and execute and market_open and not failures and status.startswith(("executed", "nothing")):
        state.update(initialized=True, last_executed_decision=d["asof"],
                     executed_at=datetime.now(timezone.utc).isoformat(),
                     last_orders=[{k: o[k] for k in ("symbol", "side", "qty", "price", "reason")} for o in executed])
        if ai_plan is not None:
            state["ai_executed"] = {"tilts": ai_plan["tilts"], "hold": ai_plan["hold"]}
        lines = ["🔁 Paper account: orders sent (may not be filled yet)"] if executed else ["✅ Paper account: no rebalance orders needed"]
        lines += [f"{'🔴 Sell' if o['side'] == 'sell' else '🟢 Buy'} {o['qty']:g} {o['symbol']} (about ${o['qty'] * o['price']:,.0f})"
                  for o in executed]
        _notify(config, "\n".join(lines), "trades")
    if not d["halted"]:
        state.pop("halt_alerted", None)
    elif not state.get("halt_alerted"):
        state["halt_alerted"] = True
        if execute:
            _notify(config, f"🚨 Buying stopped: account value fell more than "
                            f"{cfg.get('max_drawdown_halt_pct', 20)}% from its high. "
                            f"Only sell orders can run until you review.", "alerts")
    state.update(strategy_version=d["label"], asof=d["asof"], status=status,
                 gross_target=round(d["gross"], 4), overlay_note=margin_note,
                 targets={s: round(float(w), 6) for s, w in d["targets"].items()},
                 rules_targets={s: round(float(w), 6) for s, w in d["rules_targets"].items()},
                 ai_plan=ai_plan,
                 signals=d["info"]["signals"], in_market=d["info"]["in_market"],
                 tranche_last=d["info"]["tranche_last"], tranches=d["info"]["tranches"],
                 pending_plan=[{k: o[k] for k in ("symbol", "side", "qty", "price", "reason")} for o in d["orders"]]
                 if d["due"] and not status.startswith(("executed", "nothing")) else [],
                 updated_at=datetime.now(timezone.utc).isoformat())
    _write_json(STATE_PATH, state)
    print(f"  Status: {status}")

    if account:
        try:
            invested = sum(abs(float(p.get("market_value") or 0)) for p in positions
                           if p.get("symbol") != params["cash_symbol"])
            _record_equity(account, _Exposure(invested), len(holdings), d["label"])
        except Exception as e:
            logger.warning("Equity curve update failed: %s", e)
    if config.get("dashboard", {}).get("enabled", True):
        try:
            from dashboard.render import write_dashboard
            write_dashboard(config, account or None, positions if account else None)
        except Exception as e:
            logger.warning("Dashboard render failed: %s", e)
    return {"status": status, "due": d["due"], "orders": len(executed), "failures": len(failures)}


def _execute(dp, config: dict, d: dict, universe: list, closes: pd.DataFrame) -> tuple:
    executed, failures = [], []
    sells = [o for o in d["orders"] if o["side"] == "sell"]
    buys = [o for o in d["orders"] if o["side"] == "buy"]
    for batch in (sells, buys):
        for o in batch:
            sym = o["symbol"]
            try:
                if o["side"] == "sell" and o["reason"] == "exit":
                    res = dp.close_position(sym)      # also cancels its working orders (old stock stops)
                else:
                    res = dp.place_order(sym, o["qty"], o["side"], time_in_force="day")
                oid = str(res.get("id", ""))
                status = "submitted"
                executed.append(o)
            except Exception as e:
                oid, status = "", f"failed: {e}"
                failures.append({**o, "error": str(e)})
            log_trade(symbol=sym, action=o["side"], qty=o["qty"], price=o["price"], order_id=oid,
                      reason=f"{d['label']} {o['reason']} ({d['reason']})", source="etf_trend",
                      status=status if not status.startswith("failed") else "failed")
            log_audit_event(snapshot_id=f"etf-{d['asof']}", symbol=sym, strategy_name=d["label"],
                            strategy_signal=1 if o["side"] == "buy" else -1, advisor_enabled=False,
                            risk_passed=True, risk_action="allow", risk_reason=d["reason"],
                            execution_status=status if not status.startswith("failed") else "failed",
                            order_id=oid, execution_price=o["price"],
                            execution_error="" if not status.startswith("failed") else status,
                            cycle_outcome=f"rebalance_{o['reason']}")
        if batch is sells and sells and buys:
            time.sleep(float(config.get("etf_trend", {}).get("sell_settle_seconds", 2)))
    try:
        _shadow_jev(dp, config, [o for o in executed if o["reason"] == "enter"], closes, d)
    except Exception as e:  # never blocks trading
        logger.warning("Jev shadow review failed: %s", e)
    for f in failures:
        print(f"    ✗ {f['side']} {f['symbol']} failed: {f['error']}")
    if failures:
        _notify(config, "⚠️ Some paper orders failed. Bot will retry; check the dashboard. " + "; ".join(
            f"{f['symbol']} {f['side']}" for f in failures), "alerts")
    return executed, failures


def _shadow_jev(dp, config: dict, entries: list, closes: pd.DataFrame, d: dict) -> None:
    """Ask Jev about new entries AFTER they are placed; log its view as a
    forecast ([taken] if it agreed, [vetoed] if it would have blocked) so the
    decision-policy learner can score Jev without letting it touch orders."""
    jcfg = config.get("jev", {})
    if not jcfg.get("enabled", False) or not entries:
        return
    from jev import JevAgent
    from risk import RiskManager
    from snapshot import build_snapshot
    from engine import _record_signal_forecast
    jev = JevAgent(config)
    rm = RiskManager(config, initial_balance=config.get("paper_trading", {}).get("base_balance_usd", 100000))
    min_conf = float(jcfg.get("min_confidence_entry", 0.6))
    for o in entries[: int(jcfg.get("shadow_max_calls", 3))]:
        sym = o["symbol"]
        s = closes[sym].dropna()
        df = pd.DataFrame({"Open": s, "High": s, "Low": s, "Close": s, "Volume": 0.0})
        snap = build_snapshot(dp, rm, sym, 1, et.VERSION, df)
        r = jev.decide(snap)
        if not (r.success and r.data):
            continue
        agreed = r.data.action == "buy" and r.data.confidence >= min_conf
        _record_signal_forecast(config, sym, o["price"], o["price"] * 0.9, r.data, et.VERSION,
                                snap.get("snapshot_id", ""), "taken" if agreed else "vetoed")
        log_audit_event(snapshot_id=snap.get("snapshot_id", ""), symbol=sym, strategy_name=et.VERSION,
                        strategy_signal=1, advisor_enabled=False, jev_action=r.data.action,
                        jev_confidence=r.data.confidence, jev_rationale=r.data.rationale or "",
                        execution_status="not_attempted", cycle_outcome="jev_shadow")
        print(f"    Jev (shadow) on {sym}: {r.data.action} conf {r.data.confidence:.2f} "
              f"→ {'agrees' if agreed else 'would have blocked'} (not used)")


# ---------------------------------------------------------------------- #
# Replay: the live decision code against a simulated broker
# ---------------------------------------------------------------------- #

def replay(closes: pd.DataFrame, opens: pd.DataFrame, start, config: dict, cash0: float = 100_000.0,
           cost: float = 0.001) -> dict:
    """
    Walk history one session at a time. Each morning the bot sees closes up to
    the previous session (same 2-year window as live), decide() plans orders,
    and they fill at that day's open with `cost` per side; whole shares, sells
    before buys, cash buffer — exactly the live order logic. Returns the replay
    equity, the model equity (strategies.etf_trend.model_equity) and the
    tracking error between them.
    """
    params = et.params_from_config(config)
    cfg = config.get("etf_trend", {}) or {}
    c, o = _naive(closes), _naive(opens).reindex(_naive(closes).index)
    days = c.index[c.index >= pd.Timestamp(start)]
    window = 520
    hold, cash, state, peak = {}, cash0, {}, cash0
    eq, n_orders = {}, 0
    ovs = ov.settings(cfg)
    b_day = ovs["borrow_rate_pct"] / 100 / 252 if ov.active(ovs) else 0.0
    for day in days:
        if cash < 0:
            cash *= 1 + b_day
        i = c.index.get_loc(day)
        hist = c.iloc[max(0, i - window):i]
        if len(hist) < 260:
            continue
        prev_px = {s: float(hist[s].iloc[-1]) for s in hist.columns if pd.notna(hist[s].iloc[-1])}
        equity = cash + sum(q * prev_px.get(s, 0) for s, q in hold.items())
        peak = max(peak, equity)
        d = decide(hist, hold, prev_px, equity, cash, state, cfg, params, peak)
        if d["due"]:
            for od in d["orders"]:                          # sells then buys, at today's open
                px = float(o.at[day, od["symbol"]]) if pd.notna(o.at[day, od["symbol"]]) else prev_px[od["symbol"]]
                q = od["qty"]
                if od["side"] == "sell":
                    q = min(q, hold.get(od["symbol"], 0))
                    cash += q * px * (1 - cost)
                    hold[od["symbol"]] = hold.get(od["symbol"], 0) - q
                else:
                    room = cash + ov.borrow_limit(d["targets"], params["cash_symbol"]) * equity
                    q = min(q, np.floor(max(0.0, room) / (px * (1 + cost))))
                    if q <= 0:
                        continue
                    cash -= q * px * (1 + cost)
                    hold[od["symbol"]] = hold.get(od["symbol"], 0) + q
                n_orders += 1
            hold = {s: q for s, q in hold.items() if q > 0}
            state.update(initialized=True, last_executed_decision=d["asof"])
        eq[day] = cash + sum(q * float(c.at[day, s]) for s, q in hold.items())
    rep = pd.Series(eq) / cash0
    model = et.model_equity(c, o, rep.index[0], params, cost)
    model = model.reindex(rep.index).ffill()
    diff = rep.pct_change() - model.pct_change()
    return {"replay": rep, "model": model, "orders": n_orders,
            "tracking_error_pct": float(diff.std() * np.sqrt(252) * 100),
            "replay_return_pct": float((rep.iloc[-1] - 1) * 100),
            "model_return_pct": float((model.iloc[-1] / model.iloc[0] - 1) * 100)}
