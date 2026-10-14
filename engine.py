"""
Trading engine — one decision cycle shared by `run` (monitor) and `trade` (execute).

Cycle:
  0. Acquire a non-blocking lock (overlapping cron runs exit immediately)
  1. Fetch account + positions + open orders + clock ONCE
  2. Rebuild RiskManager from broker state (+ persisted peak / halt flag)
  3. Fetch bars for universe ∪ held symbols in batched requests
  4. Per symbol:
       held position  → v1.1 exit rules (deterministic, no LLM veto) →
                         optional Jev review → ratchet broker stop
       flat + fresh entry signal → research → advisor (budgeted) → Jev →
                         confidence gate → risk sizing/check → order (OTO stop)
       flat + no signal → skipped (no LLM / research calls)
  5. Reconcile positions closed outside the bot (broker stop) into the
     closed-trade ledger, persist state, print summary.

The models never place orders; the engine does, only after RiskManager
approval, and only when `execute=True` and the account is paper.
"""
from __future__ import annotations

import csv
import fcntl
import json
import logging
import os
import re
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

from data import DataProvider
from risk import RiskManager
from journal.trade_log import log_trade, ensure_header
from journal.audit_log import log_audit_event, ensure_audit_header
from advisor import AdvisorClient
from jev import JevAgent
from snapshot import build_snapshot
from schemas import ModelResult

logger = logging.getLogger("finance-bot.engine")

BOT_DIR = os.path.dirname(os.path.abspath(__file__))
JOURNAL_DIR = os.path.join(BOT_DIR, "journal")
LOCK_PATH = os.path.join(JOURNAL_DIR, "cycle.lock")
RISK_STATE_PATH = os.path.join(JOURNAL_DIR, "risk_state.json")
POSITIONS_STATE_PATH = os.path.join(JOURNAL_DIR, "positions_state.json")
ADVISOR_STATE_PATH = os.path.join(JOURNAL_DIR, ".advisor_last_call")
CLOSED_TRADES_PATH = os.path.join(JOURNAL_DIR, "closed_trades.csv")
UNIVERSE_CACHE_PATH = os.path.join(JOURNAL_DIR, "cache", "universe.json")
ACTIVE_STRATEGY_PATH = os.path.join(BOT_DIR, "learning", "active_strategy.json")
DECISION_POLICY_PATH = os.path.join(BOT_DIR, "learning", "decision_policy.json")
EQUITY_CURVE_PATH = os.path.join(JOURNAL_DIR, "equity_curve.csv")
SHADOW_CACHE_DIR = os.path.join(JOURNAL_DIR, "cache")

CLOSED_TRADE_HEADERS = [
    "closed_at", "symbol", "qty", "entry_price", "exit_price", "pnl_usd",
    "pnl_pct", "hold_days", "exit_reason", "strategy_version", "forecast_id",
]

# Symbols that are warrants/units/rights/preferreds rather than common stock
_JUNK_NAME_RE = re.compile(
    r"\b(warrants?|units?|rights?|preferred|depositary|notes?|debentures?|"
    r"subordinated|%|trust preferred|acquisition corp)\b", re.I,
)
_CLEAN_SYMBOL_RE = re.compile(r"^[A-Z]{1,5}$")


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

class CycleLocked(Exception):
    pass


@contextmanager
def cycle_lock(path: Optional[str] = None):
    """Exclusive non-blocking lock so a slow cycle never stacks with the next."""
    path = path or LOCK_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = open(path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise CycleLocked("previous cycle still running")
    try:
        fh.write(f"{os.getpid()} {datetime.now().isoformat()}\n")
        fh.flush()
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def _read_json(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


class AdvisorBudget:
    """
    Limits advisor (Kimi) calls per cycle AND per time interval.

    The old gate was evaluated once per cycle and then advisor.analyze() ran
    inside the per-symbol loop — up to one call per symbol whenever the
    interval elapsed. It also stamped the timestamp even if no call was made.
    """

    def __init__(self, config: dict, enabled: bool, state_file: Optional[str] = None):
        adv = config.get("advisor", {})
        self.interval_min = adv.get("interval_minutes", 15)
        self.max_calls = adv.get("max_calls_per_cycle", 3)
        self.state_file = state_file or ADVISOR_STATE_PATH
        self.used = 0
        self.window_open = enabled and self._interval_elapsed()

    def _interval_elapsed(self) -> bool:
        if self.interval_min <= 0:
            return True
        last = _read_json(self.state_file, {}).get("last_call", 0)
        return datetime.now().timestamp() - float(last or 0) >= self.interval_min * 60

    def take(self) -> bool:
        if not self.window_open or self.used >= self.max_calls:
            return False
        self.used += 1
        if self.used == 1:
            _write_json(self.state_file, {"last_call": datetime.now().timestamp()})
        return True


def _skipped_advisor(snapshot_id: str, why: str) -> ModelResult:
    return ModelResult(success=False, error=why, snapshot_ref=snapshot_id)


def _verdict_key(symbol: str, kind: str) -> str:
    return f"{symbol}|{kind}"


def _cached_jev_result(verdict: dict, snapshot_id: str) -> ModelResult:
    """Rebuild today's stored Jev verdict so it is reused instead of re-asked."""
    from schemas import JevDecision
    return ModelResult(success=True, snapshot_ref=snapshot_id, data=JevDecision(
        action=verdict["action"], qty=verdict.get("qty") or 0,
        stop_price=verdict.get("stop_price"), confidence=float(verdict["confidence"]),
        rationale=f"[verdict from {verdict.get('at', '?')}, not re-asked] {verdict.get('rationale', '')}"[:500]))


def _audit_fields(adv: Optional[ModelResult], jev: Optional[ModelResult]) -> dict:
    """Flatten advisor/Jev results into log_audit_event kwargs."""
    out = {}
    a = adv.data if (adv and adv.success and adv.data) else None
    out.update(
        advisor_recommendation=a.recommendation if a else None,
        advisor_confidence=a.confidence if a else None,
        advisor_evidence=a.evidence if a else "",
        advisor_risks=a.risks if a else "",
        advisor_error=(adv.error or "") if adv else "",
    )
    j = jev.data if (jev and jev.success and jev.data) else None
    out.update(
        jev_action=j.action if j else None,
        jev_qty=j.qty if j else 0,
        jev_stop_price=j.stop_price if j else None,
        jev_rationale=j.rationale if j else "",
        jev_confidence=j.confidence if j else None,
        jev_error=(jev.error or "") if jev else "",
    )
    return out


def append_closed_trade(row: dict, path: Optional[str] = None) -> None:
    """Closed-trade ledger: the outcome data the learning loop consumes."""
    path = path or CLOSED_TRADES_PATH
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CLOSED_TRADE_HEADERS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CLOSED_TRADE_HEADERS})


def read_closed_trades(path: Optional[str] = None) -> list:
    path = path or CLOSED_TRADES_PATH
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------- #
# Strategy + universe
# ---------------------------------------------------------------------- #

def strategy_from_config(config: dict):
    """
    Build the active strategy from its genome (strategies/families.py).

    Source: learning/active_strategy.json (written by `learning-approve` or
    an auto-promotion), else config.yaml `strategy` (family defaults to
    sma_cross = the v1.1 rules). Returns (strategy, version_label); the label
    is stamped on audit rows, forecasts and closed trades.
    """
    from strategies.families import make_strategy

    rec = _read_json(ACTIVE_STRATEGY_PATH, None)
    if rec and rec.get("genome"):
        genome = dict(rec["genome"])
    elif rec and rec.get("params"):  # legacy approval record
        genome = {"family": "sma_cross", **rec["params"],
                  "version": f"v1.1+{rec.get('candidate_id', 'learned')}"}
    else:
        genome = dict(config.get("strategy", {}))
        genome.setdefault("family", "sma_cross")
        if genome["family"] != "sma_cross":
            genome.pop("version", None)
    strategy = make_strategy(genome)
    return strategy, strategy.version


_STRATEGY_CACHE: dict = {}


def _strategy_for_genome(genome: dict):
    """Positions are managed by the strategy that OPENED them, even after a switch."""
    from strategies.families import make_strategy, genome_hash
    key = genome_hash(genome) + str(genome.get("version"))
    if key not in _STRATEGY_CACHE:
        _STRATEGY_CACHE[key] = make_strategy(genome)
    return _STRATEGY_CACHE[key]


def _bars_since(df, entry_time: Optional[str]) -> Optional[int]:
    try:
        t = datetime.fromisoformat(entry_time)
    except (TypeError, ValueError):
        return None
    idx = df.index
    t = t.astimezone(timezone.utc)
    if getattr(idx, "tz", None) is None:
        t = t.replace(tzinfo=None)
    return int((idx > t).sum())


def _record_equity(account: dict, risk_mgr, n_positions: int, version: str) -> None:
    """One row per day (last cycle wins): the ACTUAL paper track record."""
    import csv as _csv
    today = datetime.now(timezone.utc).date().isoformat()
    rows = []
    if os.path.exists(EQUITY_CURVE_PATH):
        with open(EQUITY_CURVE_PATH) as f:
            rows = [r for r in _csv.DictReader(f) if r.get("date") != today]
    rows.append({"date": today, "equity": account.get("equity") or account.get("portfolio_value"),
                 "last_equity": account.get("last_equity", ""),
                 "exposure": round(risk_mgr.total_open_exposure(), 2), "positions": n_positions,
                 "strategy_version": version, "updated_at": datetime.now(timezone.utc).isoformat()})
    os.makedirs(os.path.dirname(EQUITY_CURVE_PATH), exist_ok=True)
    with open(EQUITY_CURVE_PATH, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
        w.writeheader()
        w.writerows(rows)


def resolve_market_universe(config: dict, dp: DataProvider) -> list:
    """
    Resolve the market universe.

    mode="static":  the explicit symbol list.
    mode="dynamic": Alpaca assets → common-stock filter → batched snapshot
                    liquidity screen (min_price, min_dollar_volume) → ranked by
                    dollar volume → top `max_symbols`. Cached for the day so
                    the 30k-asset fetch does not run every 5 minutes.

    Falls back to the static list if dynamic resolution fails.
    """
    ucfg = config.get("market_universe", {})
    static = ucfg.get("symbols", ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"])
    if ucfg.get("mode", "static") == "static":
        return list(static)

    dyn = ucfg.get("dynamic", {})
    max_symbols = int(dyn.get("max_symbols", 50) or 0)
    hard_cap = int(dyn.get("hard_cap", 200))
    if max_symbols <= 0 or max_symbols > hard_cap:
        logger.warning("max_symbols=%s is unbounded/too large; capping at %d", max_symbols, hard_cap)
        max_symbols = hard_cap

    today = datetime.now().strftime("%Y-%m-%d")
    cache = _read_json(UNIVERSE_CACHE_PATH, {})
    cache_key = json.dumps(dyn, sort_keys=True)
    if cache.get("date") == today and cache.get("key") == cache_key and cache.get("symbols"):
        return cache["symbols"][:max_symbols]

    try:
        assets = dp.get_all_assets_meta()
    except Exception as e:
        logger.error("Failed to fetch dynamic universe: %s. Falling back to static.", e)
        return list(static)

    asset_class = dyn.get("asset_class", "us_equity")
    exchanges = set(dyn.get("exchanges", ["NYSE", "NASDAQ", "ARCA", "AMEX", "BATS"]))
    exclude = set(dyn.get("exclude", []))
    candidates = []
    for a in assets:
        sym = a.get("symbol") or ""
        if not a.get("tradable") or sym in exclude:
            continue
        if a.get("status") not in (None, "active"):
            continue
        if asset_class != "all" and a.get("asset_class") != asset_class:
            continue
        if a.get("exchange") and a["exchange"] not in exchanges:
            continue
        if not _CLEAN_SYMBOL_RE.match(sym) or _JUNK_NAME_RE.search(a.get("name", "")):
            continue  # warrants (.WS / ...W), units, preferreds (.PRA), rights
        candidates.append(sym)

    min_price = float(dyn.get("min_price", 5.0))
    min_dv = float(dyn.get("min_dollar_volume", 20_000_000))
    try:
        snaps = dp.get_snapshots(candidates)
    except Exception as e:
        logger.error("Liquidity screen failed: %s. Falling back to static.", e)
        return list(static)

    ranked = []
    for sym, sn in snaps.items():
        price = sn["price"]
        # previous full session is a better liquidity measure than a partial day
        dollar_vol = sn["prev_close"] * max(sn["prev_volume"], sn["volume"])
        if price >= min_price and dollar_vol >= min_dv:
            ranked.append((dollar_vol, sym))
    ranked.sort(reverse=True)
    symbols = [s for _, s in ranked]
    if not symbols:
        logger.error("Liquidity screen returned nothing; falling back to static list")
        return list(static)

    _write_json(UNIVERSE_CACHE_PATH, {"date": today, "key": cache_key, "symbols": symbols[:hard_cap]})
    logger.info(
        "Dynamic universe: %d assets → %d common stocks → %d liquid → using top %d",
        len(assets), len(candidates), len(symbols), min(max_symbols, len(symbols)),
    )
    return symbols[:max_symbols]


def fetch_bars(dp: DataProvider, symbols: list, lookback_days: int) -> tuple[dict, dict]:
    """Batched Alpaca bars with per-symbol yfinance fallback. Returns (bars, sources)."""
    bars, sources = {}, {}
    if dp.data_client:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        bars = dp.fetch_alpaca_bars_multi(symbols, start, end)
        sources = {s: "alpaca" for s in bars}
    # yfinance fallback must honour the requested history (previously >180d
    # always fetched "1y", silently truncating 2-5y learning requests)
    period = next(p for d, p in ((180, "6mo"), (365, "1y"), (730, "2y"), (1825, "5y"), (10**9, "max"))
                  if lookback_days <= d)
    for sym in symbols:
        if sym in bars and not bars[sym].empty:
            continue
        df = dp.fetch_yf_history(sym, period=period, interval="1d")
        if not df.empty:
            bars[sym] = df
            sources[sym] = "yfinance"
    return bars, sources


# ---------------------------------------------------------------------- #
# The cycle
# ---------------------------------------------------------------------- #

def run_cycle(dp: DataProvider, config: dict, execute: bool) -> dict:
    """Run one decision cycle. Returns a summary dict."""
    try:
        with cycle_lock():
            return _run_cycle_locked(dp, config, execute)
    except CycleLocked:
        print("[SKIPPED] Previous cycle still running (lock held)")
        return {"skipped": "locked"}


def _run_cycle_locked(dp: DataProvider, config: dict, execute: bool) -> dict:
    exec_cfg = config.get("execution", {})
    jev_cfg = config.get("jev", {})
    min_conf_entry = float(jev_cfg.get("min_confidence_entry", 0.6))
    min_conf_exit = float(jev_cfg.get("min_confidence_exit", 0.6))
    review_positions = bool(jev_cfg.get("review_open_positions", True))
    require_open = bool(exec_cfg.get("require_market_open", True))
    entry_lookback = int(config.get("strategy", {}).get("entry_lookback_bars", 3))
    log_no_signal = bool(config.get("journal", {}).get("log_no_signal", False))
    # Jev is stochastic (same snapshot → conf 0.04..0.67 on 2026-10-05). Asking it
    # again every 5-min cycle while a signal stays fresh turns the confidence
    # gate into "retry until yes". One verdict per symbol per day instead.
    once_per_day = bool(jev_cfg.get("decide_once_per_day", True))
    verdicts_path = os.path.join(SHADOW_CACHE_DIR,
                                 f"jev_verdicts_{datetime.now(timezone.utc).date().isoformat()}.json")
    verdicts = _read_json(verdicts_path, {}) if once_per_day else {}

    def remember(symbol: str, kind: str, jev_r: ModelResult) -> None:
        if not once_per_day or not (jev_r.success and jev_r.data):
            return  # failed calls are retried next cycle
        d = jev_r.data
        verdicts[_verdict_key(symbol, kind)] = {
            "action": d.action, "confidence": d.confidence, "qty": d.qty,
            "stop_price": d.stop_price, "rationale": d.rationale,
            "at": datetime.now(timezone.utc).strftime("%H:%M UTC")}
        _write_json(verdicts_path, verdicts)

    if execute and not dp.trading_client:
        print("ERROR: Alpaca trading client not configured. Cannot trade.")
        return {"error": "no_trading_client"}
    if execute and not dp.paper:
        print("ERROR: refusing to execute — account is not paper (APCA_PAPER!=true).")
        return {"error": "not_paper"}

    if config.get("learning", {}).get("enabled") and \
            config.get("learning", {}).get("promotion", {}).get("allow_auto_promotion"):
        try:
            from learning.loop import apply_pending_promotion
            switched = apply_pending_promotion(config)
            if switched:
                print(f"  ✅ Applied announced strategy switch → {switched}")
        except Exception as e:
            logger.warning("Pending promotion check failed: %s", e)

    policy = _read_json(DECISION_POLICY_PATH, {})
    if "min_confidence_entry" in policy:
        # Learned from realised outcomes of taken vs vetoed signals
        min_conf_entry = max(0.0, min(0.9, float(policy["min_confidence_entry"])))

    strategy, strat_version = strategy_from_config(config)
    ensure_header()
    ensure_audit_header()

    # --- 1. Broker state, fetched ONCE ---
    account, positions, open_orders, market_open = {}, [], [], False
    if dp.trading_client:
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

    held = {p["symbol"]: p for p in positions if float(p.get("qty", 0) or 0) > 0}
    orders_by_symbol: dict = {}
    for o in open_orders:
        orders_by_symbol.setdefault(o.get("symbol"), []).append(o)

    # --- 2. Risk state rebuilt from broker + persisted peak/halt ---
    balance = config.get("paper_trading", {}).get("base_balance_usd", 100000)
    risk_mgr = RiskManager(config, initial_balance=balance)
    risk_mgr.load_state(RISK_STATE_PATH)
    if account:
        risk_mgr.sync_from_broker(account, positions)
    was_halted = risk_mgr.halted

    pos_state = _read_json(POSITIONS_STATE_PATH, {})

    # --- 3. Universe + bars ---
    universe = resolve_market_universe(config, dp)
    # Held symbols are ALWAYS managed, even if they dropped out of the universe
    symbols = list(dict.fromkeys(list(held) + universe))
    need_bars = [strategy.min_bars] + [
        _strategy_for_genome(st["genome"]).min_bars for sym, st in pos_state.items()
        if sym in held and st.get("genome")]
    lookback = max(int(config.get("data", {}).get("lookback_days", 150)),
                   int(max(need_bars) * 1.6) + 20)
    bars, sources = fetch_bars(dp, symbols, lookback)

    advisor = AdvisorClient(config)
    jev = JevAgent(config)
    budget = AdvisorBudget(config, advisor.enabled)
    aggregator = None
    if config.get("research", {}).get("enabled", False):
        from research.roles import ResearchAggregator
        aggregator = ResearchAggregator(config)
        aggregator.set_data_provider(dp)

    mode = "TRADE (paper)" if execute else "MONITOR (no orders)"
    print(f"=== Cycle [{mode}] {datetime.now().isoformat()} ===")
    print(f"  Strategy: {strategy.name} [{strat_version}] sma={strategy.sma_short}/{strategy.sma_long}")
    print(f"  Universe: {len(universe)} symbols, held: {len(held)}, market_open={market_open}")
    print(f"  Advisor: {advisor.provider}/{advisor.model} enabled={advisor.enabled} "
          f"window_open={budget.window_open} max_calls={budget.max_calls}")
    print(f"  Jev: {jev.provider}/{jev.model} min_conf entry={min_conf_entry} exit={min_conf_exit}")
    print(f"  Equity: ${risk_mgr.balance:,.2f} peak ${risk_mgr.peak_balance:,.2f} "
          f"exposure ${risk_mgr.total_open_exposure():,.2f} halted={risk_mgr.halted}")

    submitted_entries = set()
    counts = {"no_data": 0, "no_signal": 0, "entries_considered": 0, "orders": 0,
              "exits": 0, "holds": 0, "rejected": 0}

    def research_for(snap, df):
        if not aggregator:
            return None
        try:
            return aggregator.run_full_analysis(snap, df)["findings"]
        except Exception as e:
            logger.warning("Research roles failed for %s: %s", snap["symbol"], e)
            return None

    def advise(snap):
        if not advisor.enabled:
            return None
        if budget.take():
            return advisor.analyze(snap)
        return _skipped_advisor(snap["snapshot_id"], "Advisor not called this cycle (interval/budget)")

    for symbol in symbols:
        df = bars.get(symbol)
        pos = held.get(symbol)
        st_prev = pos_state.get(symbol, {})
        sym_strategy = _strategy_for_genome(st_prev["genome"]) if (pos and st_prev.get("genome")) else strategy
        if df is None or df.empty or len(df) < sym_strategy.min_bars:
            counts["no_data"] += 1
            if symbol in held:
                print(f"  {symbol}: HELD but insufficient data ({0 if df is None else len(df)} bars) — broker stop still protects")
            continue

        source = sources.get(symbol, "?")
        close = float(df["Close"].iloc[-1])
        signal = sym_strategy.live_signal(df, entry_lookback)

        if pos is None and signal != 1:
            counts["no_signal"] += 1
            if log_no_signal:
                log_audit_event(snapshot_id="", symbol=symbol, strategy_name=strat_version,
                                strategy_signal=signal, advisor_enabled=advisor.enabled,
                                execution_status="not_attempted", cycle_outcome="no_signal")
            continue

        snap = build_snapshot(
            dp=dp, risk_mgr=risk_mgr, symbol=symbol, strategy_signal=signal,
            strategy_name=f"{strategy.name} [{strat_version}]", df=df,
            sma_short=strategy.sma_short, sma_long=strategy.sma_long,
            account_state=account or None, positions=positions,
        )
        base = dict(snapshot_id=snap["snapshot_id"], symbol=symbol,
                    strategy_name=strat_version, strategy_signal=signal,
                    advisor_enabled=advisor.enabled)

        # ============ Held position: exits + stop management ============
        if pos is not None:
            held_qty = float(pos["qty"])
            entry = float(pos.get("avg_entry_price") or close)
            st = pos_state.setdefault(symbol, {
                "entry_price": entry, "entry_time": datetime.now(timezone.utc).isoformat(),
                "strategy_version": strat_version, "forecast_id": "",
            })
            st["confirmed"] = True
            st["entry_price"] = entry
            ev = sym_strategy.evaluate_exit(df, entry, st.get("stop"), _bars_since(df, st.get("entry_time")))
            st["stop"] = ev["stop"]
            adv_r = jev_r = None
            exit_reason = ev["reason"] if ev["exit"] else ""

            if not ev["exit"] and review_positions:
                prior = verdicts.get(_verdict_key(symbol, "review"))
                if prior:
                    jev_r = _cached_jev_result(prior, snap["snapshot_id"])
                else:
                    findings = research_for(snap, df)
                    adv_r = advise(snap)
                    jev_r = jev.decide(snap, adv_r, findings)
                    remember(symbol, "review", jev_r)
                if (jev_r.success and jev_r.data and jev_r.data.action == "sell"
                        and jev_r.data.confidence >= min_conf_exit):
                    exit_reason = f"jev_sell_conf_{jev_r.data.confidence:.2f}"

            print(f"  {symbol} [{source}] HELD {held_qty:g} @ {entry:.2f} close={close:.2f} "
                  f"stop={ev['stop']:.2f} signal={signal}"
                  + (f" → EXIT ({exit_reason})" if exit_reason else " → hold"))

            if not exit_reason:
                counts["holds"] += 1
                # Ratchet the broker-side stop up to the new v1.1 stop
                if execute and market_open:
                    for o in orders_by_symbol.get(symbol, []):
                        if o.get("side") == "sell" and o.get("type") in ("stop", "stop_limit"):
                            cur = float(o.get("stop_price") or 0)
                            if ev["stop"] > cur + 0.01:
                                try:
                                    dp.replace_stop(o["id"], ev["stop"])
                                    print(f"    stop ratcheted {cur:.2f} → {ev['stop']:.2f}")
                                except Exception as e:
                                    logger.warning("Stop ratchet failed for %s: %s", symbol, e)
                log_audit_event(**base, **_audit_fields(adv_r, jev_r),
                                execution_status="not_attempted", cycle_outcome="hold_position")
                continue

            check = risk_mgr.check_trade(symbol, "sell", held_qty, close, None, held_qty=held_qty)
            status, order_id, err, outcome = "not_attempted", "", "", "would_exit"
            if execute:
                if require_open and not market_open:
                    status, outcome = "skipped", "market_closed"
                else:
                    try:
                        order = dp.close_position(symbol)
                        order_id = str(order.get("id", ""))
                        status, outcome = "submitted", "position_closed"
                        counts["exits"] += 1
                        pnl = (close - entry) * held_qty
                        append_closed_trade({
                            "closed_at": datetime.now(timezone.utc).isoformat(),
                            "symbol": symbol, "qty": held_qty, "entry_price": round(entry, 4),
                            "exit_price": round(close, 4), "pnl_usd": round(pnl, 2),
                            "pnl_pct": round((close / entry - 1) * 100, 3) if entry else 0,
                            "hold_days": _hold_days(st.get("entry_time")),
                            "exit_reason": exit_reason,
                            "strategy_version": st.get("strategy_version", strat_version),
                            "forecast_id": st.get("forecast_id", ""),
                        })
                        # day P&L already reflects this via broker equity; only free exposure
                        risk_mgr.release_position(symbol)
                        pos_state.pop(symbol, None)
                        _notify(config, f"🔴 Paper sell sent: {held_qty:g} {symbol} (about ${close:,.2f} each). "
                                        f"Estimated change: ${pnl:+,.0f}. Check final fill on the dashboard.",
                                "trades")
                    except Exception as e:
                        status, outcome, err = "failed", "execution_failed", str(e)
                        print(f"    ✗ EXIT FAILED — {e}")
            log_audit_event(**base, **_audit_fields(adv_r, jev_r),
                            risk_passed=check.passed, risk_action=check.action,
                            risk_reason=f"{check.reason}; exit={exit_reason}",
                            execution_status=status, order_id=order_id,
                            execution_price=close, execution_error=err, cycle_outcome=outcome)
            log_trade(symbol=symbol, action="sell", qty=held_qty, price=close, order_id=order_id,
                      reason=f"exit={exit_reason}", source=source, status=status)
            continue

        # ============ Flat + fresh entry signal ============
        counts["entries_considered"] += 1
        if risk_mgr.halted:
            print(f"  {symbol}: entry signal but trading HALTED — {risk_mgr.halt_reason}")
            log_audit_event(**base, risk_passed=False, risk_action="halt",
                            risk_reason=risk_mgr.halt_reason or "", execution_status="skipped",
                            cycle_outcome="halted")
            continue
        if any(o.get("side") == "buy" for o in orders_by_symbol.get(symbol, [])):
            print(f"  {symbol}: entry signal but a buy order is already working — skip")
            continue
        if risk_mgr.available_exposure_capacity() < risk_mgr.limits.get("min_order_collateral_usd", 15.0):
            print(f"  {symbol}: entry signal but no exposure capacity left — skip (no model calls)")
            counts["rejected"] += 1
            continue

        prior = verdicts.get(_verdict_key(symbol, "entry"))
        if prior and (prior["action"] != "buy" or prior["confidence"] < min_conf_entry) \
                and policy.get("jev_can_veto") is not False:
            print(f"  {symbol}: entry signal — Jev already declined today "
                  f"({prior['action']} conf {prior['confidence']:.2f} at {prior.get('at')}); not re-asking")
            continue
        if prior:
            # Approved earlier but not executed (market closed / risk / order failure)
            adv_r, jev_r = None, _cached_jev_result(prior, snap["snapshot_id"])
        else:
            findings = research_for(snap, df)
            adv_r = advise(snap)
            jev_r = jev.decide(snap, adv_r, findings)
            remember(symbol, "entry", jev_r)
        fields = _audit_fields(adv_r, jev_r)

        if not jev_r.success or not jev_r.data:
            print(f"  {symbol}: entry signal — Jev FAILED ({jev_r.error}) → no trade")
            log_audit_event(**base, **fields, execution_status="not_attempted", cycle_outcome="jev_failed")
            continue
        d = jev_r.data
        jev_overridden = False
        if policy.get("jev_can_veto") is False and (d.action != "buy" or d.confidence < min_conf_entry):
            # Learned policy: Jev's vetoes were anti-predictive → follow the
            # strategy signal; keep logging what Jev would have done.
            from schemas import JevDecision
            d = JevDecision(action="buy", qty=0, stop_price=None, confidence=d.confidence,
                            rationale=f"Jev veto disabled by learned policy (Jev said {d.action} "
                                      f"conf {d.confidence:.2f})")
            jev_overridden = True
        if d.action != "buy" or (d.confidence < min_conf_entry and not jev_overridden):
            why = "hold" if d.action != "buy" else f"low confidence {d.confidence:.2f} < {min_conf_entry}"
            # Shadow forecast: lets the learning loop measure whether Jev's
            # vetoes actually avoid losers (decision-policy learning).
            _record_signal_forecast(config, symbol, close, strategy.initial_stop(df), d,
                                    strat_version, snap["snapshot_id"], "vetoed")
            print(f"  {symbol} [{source}] close={close:.2f} signal=1 → Jev {d.action} ({why})")
            counts["holds"] += 1
            log_audit_event(**base, **fields, execution_status="not_attempted",
                            cycle_outcome="hold" if d.action != "buy" else "low_confidence")
            continue

        # Stop: the strategy's own initial stop — the one the backtest and the
        # learning loop validated. Jev's stop is only a fallback: the TypeSafe
        # adapter emits a fixed close x 0.97, ~2x tighter than 3xATR.
        stop = strategy.initial_stop(df)
        if stop is None or stop >= close:
            stop = d.stop_price if (d.stop_price and close * 0.75 < d.stop_price < close) else None
        if stop is None:
            stop = round(close * 0.95, 2)
        qty = risk_mgr.size_position(close, stop)
        if d.qty and d.qty > 0:
            qty = min(qty, int(d.qty))
        check = risk_mgr.check_trade(symbol, "buy", qty, close, stop)
        if not check.passed:
            _record_signal_forecast(config, symbol, close, stop, d, strat_version,
                                    snap["snapshot_id"], "risk_rejected")
            print(f"  {symbol}: Jev buy conf={d.confidence:.2f} → ✗ RISK {check.action.upper()} — {check.reason}")
            counts["rejected"] += 1
            log_audit_event(**base, **fields, risk_passed=False, risk_action=check.action,
                            risk_reason=check.reason, execution_status="rejected",
                            cycle_outcome="risk_rejected")
            continue

        status, order_id, err, outcome = "not_attempted", "", "", "would_trade"
        if execute:
            if require_open and not market_open:
                status, outcome = "skipped", "market_closed"
            else:
                try:
                    order = dp.place_order(symbol, qty, "buy", stop_loss_price=stop)
                    order_id = str(order.get("id", ""))
                    status, outcome = "submitted", "trade_opened"
                    counts["orders"] += 1
                except Exception as e:
                    status, outcome, err = "failed", "execution_failed", str(e)
        # Count exposure even in monitor mode so later symbols see the cap
        if status in ("submitted", "not_attempted"):
            risk_mgr.record_position(symbol, qty * close)

        forecast_id = _record_signal_forecast(
            config, symbol, close, stop, d, strat_version, snap["snapshot_id"],
            ("vetoed" if jev_overridden else "taken") if status in ("submitted", "not_attempted")
            else "skipped")
        if status == "submitted":
            submitted_entries.add(symbol)
            pos_state[symbol] = {
                "entry_price": close, "entry_time": datetime.now(timezone.utc).isoformat(),
                "stop": stop, "strategy_version": strat_version, "order_id": order_id,
                "forecast_id": forecast_id, "confirmed": False,
                "genome": {**strategy.genome, "version": strat_version},
            }
            _notify(config, f"🟢 Paper buy sent: {qty} {symbol} (about ${close:,.2f} each). "
                            f"Protective sell set near ${stop:,.2f}. Check final fill on the dashboard.",
                    "trades")

        print(f"  {symbol} [{source}] close={close:.2f} signal=1 → BUY {qty} stop={stop:.2f} "
              f"conf={d.confidence:.2f} [{status}{' ' + err if err else ''}]")
        fields.update(jev_qty=qty, jev_stop_price=stop)  # what was actually sent
        log_audit_event(**base, **fields,
                        risk_passed=True, risk_action=check.action, risk_reason=check.reason,
                        execution_status=status, order_id=order_id, execution_price=close,
                        execution_error=err, cycle_outcome=outcome)
        log_trade(symbol=symbol, action="buy", qty=qty, price=close, order_id=order_id,
                  reason=f"signal=1 jev=buy conf={d.confidence:.2f} risk=allow",
                  source=source, status=status)

    # --- 5. Reconcile positions that disappeared without the bot selling ---
    pending_buys = {sym for sym, os_ in orders_by_symbol.items() if any(o.get("side") == "buy" for o in os_)}
    for sym in (list(pos_state) if account else []):
        if sym in held or sym in pending_buys or sym in submitted_entries:
            continue
        st = pos_state.pop(sym)
        if not st.get("confirmed"):
            print(f"  {sym}: entry order never filled — dropping state")
            continue
        df = bars.get(sym)
        exit_px = float(df["Close"].iloc[-1]) if df is not None and not df.empty else float(st.get("stop") or 0)
        entry = float(st.get("entry_price") or 0)
        print(f"  {sym}: position closed outside the bot (broker stop?) ~{exit_px:.2f}")
        if execute and entry:
            append_closed_trade({
                "closed_at": datetime.now(timezone.utc).isoformat(), "symbol": sym,
                "qty": "", "entry_price": entry, "exit_price": exit_px,
                "pnl_usd": "", "pnl_pct": round((exit_px / entry - 1) * 100, 3),
                "hold_days": _hold_days(st.get("entry_time")), "exit_reason": "broker_stop_or_external",
                "strategy_version": st.get("strategy_version", ""), "forecast_id": st.get("forecast_id", ""),
            })

    _write_json(POSITIONS_STATE_PATH, pos_state)
    risk_mgr.save_state(RISK_STATE_PATH)
    if account:
        try:
            _record_equity(account, risk_mgr, len(held), strat_version)
        except Exception as e:
            logger.warning("Equity curve update failed: %s", e)
    if config.get("dashboard", {}).get("enabled", True):
        try:
            from dashboard.render import write_dashboard
            write_dashboard(config, account or None, positions if account else None)
        except Exception as e:
            logger.warning("Dashboard render failed: %s", e)
    if execute and risk_mgr.halted and not was_halted:
        _notify(config, f"🚨 New buys stopped: {risk_mgr.halt_reason}. "
                        f"Sells may still run. Please review before restarting.", "alerts")

    print(f"\n  Summary: {counts}")
    print(f"  Advisor calls this cycle: {budget.used}")
    summary = risk_mgr.daily_summary()
    print(f"  Day P&L: ${summary['realized_pl']:.2f}  exposure: ${summary['total_exposure']:.2f}")
    if summary["halted"]:
        print(f"  ⚠ TRADING HALTED: {risk_mgr.halt_reason}")
    return counts


def _hold_days(entry_time: Optional[str]) -> str:
    try:
        dt = datetime.fromisoformat(entry_time)
        return str((datetime.now(timezone.utc) - dt).days)
    except (TypeError, ValueError):
        return ""


def _notify(config: dict, text: str, kind: str) -> None:
    try:
        import notify
        notify.send(config, text, kind=kind)
    except Exception as e:
        logger.warning("Notification failed: %s", e)


def _record_signal_forecast(config, symbol, close, stop, decision, strat_version,
                            snapshot_id, tag: str) -> str:
    """
    Record an entry signal as an immutable forecast, tagged with what the bot
    did with it ([taken] / [vetoed] / [risk_rejected] / [skipped]).

    Taken signals measure live calibration; vetoed ones are the
    counterfactual that tells the learning loop whether Jev's filter helps.
    De-duplicated to one forecast per symbol, tag and day (a signal stays
    live for several cycles).
    """
    try:
        from learning.forecasts import Forecast, ForecastStore, generate_forecast_id
        day = datetime.now(timezone.utc).date().isoformat()
        seen_path = os.path.join(SHADOW_CACHE_DIR, f"signals_{day}.json")
        seen = _read_json(seen_path, {})
        key = f"{symbol}|{tag}"
        if key in seen:
            return seen[key]
        fc = config.get("learning", {}).get("forecasts", {})
        horizon = int(fc.get("horizon_days", 10))
        reward_risk = float(fc.get("reward_risk", 1.5))
        if not stop or stop >= close:
            stop = close * 0.95
        decision_time = datetime.now(timezone.utc).isoformat()
        thesis = (f"[{tag}] {strat_version} entry; stop {stop:.2f}; jev={decision.action}; "
                  + (decision.rationale or ""))[:400]
        f = Forecast(
            forecast_id=generate_forecast_id(symbol, decision_time, thesis),
            decision_time=decision_time, symbol=symbol, horizon_days=horizon,
            expected_return_pct=round((close - stop) / close * 100 * reward_risk, 4),
            confidence=float(decision.confidence), thesis=thesis,
            source_publication_times="[]", strategy_version=strat_version,
            data_snapshot_id=snapshot_id,
        )
        ForecastStore(os.path.join(BOT_DIR, "learning", "forecasts.csv")).record(f)
        seen[key] = f.forecast_id
        _write_json(seen_path, seen)
        return f.forecast_id
    except Exception as e:
        logger.warning("Forecast record failed for %s: %s", symbol, e)
        return ""
