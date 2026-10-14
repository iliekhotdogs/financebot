"""ETF trend mode: strategy rules, order planning, live cycle (fake broker), replay, readiness."""
import os
import sys
import csv
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine  # noqa: E402
import engine_etf  # noqa: E402
from strategies import etf_trend as et  # noqa: E402

ET = ZoneInfo("America/New_York")
SYMS = et.DEFAULTS["symbols"]
BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_prices(end="2026-10-02", n=600, seed=1, up=("SPY", "IEF", "GLD", "EFA", "TLT")):
    idx = pd.bdate_range(end=end, periods=n)
    rng = np.random.default_rng(seed)
    data = {}
    for s in SYMS:
        drift = 0.0008 if s in up else -0.0008
        data[s] = 100 * np.cumprod(1 + rng.normal(drift, 0.008, n))
    data["BIL"] = 100 * np.cumprod(np.full(n, 1 + 0.04 / 252))
    closes = pd.DataFrame(data, index=idx)
    opens = closes.shift(1).bfill() * (1 + rng.normal(0, 0.002, closes.shape))
    return closes, opens


class FakeDP:
    paper = True
    data_client = None

    def __init__(self, closes, opens, positions=None, equity=100_000.0, cash=None, market_open=True,
                 open_orders=None):
        self.trading_client = object()
        self.closes, self.opens = closes, opens
        self.positions = positions or []
        self.equity = equity
        self.cash = equity if cash is None else cash
        self.market_open = market_open
        self.open_orders = open_orders or []
        self.orders, self.closed = [], []

    def fetch_yf_history(self, symbol, period="2y", interval="1d"):
        if symbol not in self.closes:
            return pd.DataFrame()
        return pd.DataFrame({"Open": self.opens[symbol], "Close": self.closes[symbol]})

    def get_account(self):
        return {"equity": str(self.equity), "last_equity": str(self.equity), "cash": str(self.cash),
                "portfolio_value": str(self.equity)}

    def get_positions(self):
        return self.positions

    def get_open_orders(self, symbol=None):
        return self.open_orders

    def get_clock(self):
        return {"is_open": self.market_open}

    def place_order(self, symbol, qty, side="buy", time_in_force="day", stop_loss_price=None):
        self.orders.append({"symbol": symbol, "qty": qty, "side": side, "tif": time_in_force})
        return {"id": f"ord-{len(self.orders)}"}

    def close_position(self, symbol):
        self.closed.append(symbol)
        return {"id": f"close-{symbol}"}


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    with open(os.path.join(BOT_DIR, "config.yaml")) as f:
        config = yaml.safe_load(f)
    config["dashboard"] = {"enabled": False}
    config["jev"]["enabled"] = False
    config["etf_trend"].setdefault("ai", {})["enabled"] = False   # rule tests; AI tests turn it on
    monkeypatch.setattr(engine_etf, "STATE_PATH", str(tmp_path / "etf_state.json"))
    monkeypatch.setattr(engine_etf, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(engine_etf, "EQUITY_CURVE_PATH", str(tmp_path / "equity_curve.csv"))
    monkeypatch.setattr(engine, "EQUITY_CURVE_PATH", str(tmp_path / "equity_curve.csv"))
    monkeypatch.setattr(engine, "LOCK_PATH", str(tmp_path / "cycle.lock"))
    monkeypatch.setattr(engine_etf.time, "sleep", lambda s: None)
    monkeypatch.setenv("FINBOT_NOTIFY", "off")
    import journal.trade_log as tl
    import journal.audit_log as al
    monkeypatch.setattr(tl, "LOG_PATH", str(tmp_path / "trade_log.csv"))
    monkeypatch.setattr(al, "LOG_PATH", str(tmp_path / "audit_log.csv"))
    return config


NOW = datetime(2026, 10, 5, 10, 0, tzinfo=ET)   # Monday 10:00 ET; last completed session Fri Oct 2


# ---------------------------------------------------------------- strategy

def test_config_selects_etf_mode():
    with open(os.path.join(BOT_DIR, "config.yaml")) as f:
        c = yaml.safe_load(f)
    import main
    assert main.etf_mode(c)
    assert et.params_from_config(c)["tranche_days"] == [0, 5, 10, 15]


def test_day_of_month_counts_trading_days():
    idx = pd.bdate_range("2026-09-28", "2026-10-09")
    dom = et.day_of_month(idx)
    assert dom.loc["2026-09-30"] == 2       # Mon 28, Tue 29, Wed 30
    assert dom.loc["2026-10-01"] == 0       # new month restarts
    assert dom.loc["2026-10-08"] == 5


def _monthly_path(levels):
    """Daily closes that sit at levels[m] for the whole of month m (2 years of warm-up at 100)."""
    idx = pd.bdate_range("2024-01-01", periods=21 * (12 + len(levels)), freq="B")
    months = idx.to_period("M")
    uniq = list(dict.fromkeys(months))
    lv = {m: 100.0 for m in uniq}
    for m, v in zip(uniq[-len(levels):], levels):
        lv[m] = v
    return pd.DataFrame({"SPY": [lv[m] for m in months]}, index=idx)


@pytest.mark.parametrize("levels,expected", [
    ([100.5], 0),               # within +1% buffer → stays out
    ([101.5], 1),               # above buffer → in
    ([101.5, 100.0], 1),        # back inside the buffer → stays in
    ([101.5, 98.0], 0),         # below −1% → out
])
def test_buffer_hysteresis(levels, expected):
    closes = _monthly_path(levels)
    p = {"symbols": ["SPY"], "tranche_days": [0], "lookback_months": 10}
    book = et.TrendBook(closes, p)
    for d, k in book.dom.items():
        if k == 0:
            book.rebalance(d)
    assert book.state[0]["SPY"] == expected


def test_targets_are_valid_weights():
    closes, _ = make_prices()
    w, info = et.targets_asof(closes)
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()
    assert (w.drop("BIL") <= et.DEFAULTS["max_weight"] + 1e-9).all()
    assert info["tranches"] == 4 and "BIL" in w.index


def test_targets_have_no_lookahead():
    closes, _ = make_prices()
    d = closes.index[-60]
    a, _ = et.targets_asof(closes, d)
    future = closes.copy()
    future.loc[future.index > d] *= 0.5                     # crash AFTER d
    b, _ = et.targets_asof(future, d)
    pd.testing.assert_series_equal(a, b)


def test_full_history_book_matches_point_in_time_targets():
    """The backtest/model feed TrendBook the WHOLE history; it must give the same
    weights at d as the live path, which only sees closes up to d."""
    closes, _ = make_prices()
    book = et.TrendBook(closes, et.DEFAULTS)
    checkpoints = list(closes.index[-150::7])
    j = 0
    for day, k in book.dom.items():
        if int(k) in et.DEFAULTS["tranche_days"]:
            book.rebalance(day)
        while j < len(checkpoints) and checkpoints[j] == day:
            live, _ = et.targets_asof(closes, day)
            pd.testing.assert_series_equal(book.combined().sort_index(), live.sort_index(), atol=1e-12)
            j += 1
    assert j == len(checkpoints)


def test_uptrending_assets_are_held_downtrending_are_not():
    closes, _ = make_prices()
    w, _ = et.targets_asof(closes)
    assert w.get("SPY", 0) > 0.03 and w.get("EEM", 0) == 0 and w.get("DBC", 0) == 0


def test_plan_orders_exits_trims_and_fits_cash():
    targets = pd.Series({"SPY": 0.5, "IEF": 0.3, "BIL": 0.2})
    holdings = {"TXN": 17, "SPY": 10, "GLD": 5}
    prices = {"TXN": 290.0, "SPY": 600.0, "IEF": 95.0, "BIL": 91.5, "GLD": 300.0}
    orders = et.plan_orders(targets, holdings, prices, equity=100_000, cash=84_570)
    by = {(o["symbol"], o["side"]): o for o in orders}
    assert by[("TXN", "sell")]["qty"] == 17 and by[("TXN", "sell")]["reason"] == "exit"
    assert by[("GLD", "sell")]["reason"] == "exit"
    assert [o["side"] for o in orders] == sorted([o["side"] for o in orders], key=lambda s: s != "sell")
    assert all(float(o["qty"]).is_integer() for o in orders)
    spent = sum(o["qty"] * o["price"] for o in orders if o["side"] == "buy")
    got = sum(o["qty"] * o["price"] for o in orders if o["side"] == "sell")
    assert spent <= 84_570 + got - 1_000 + 1e-6               # 1% cash buffer kept


def test_plan_orders_skips_tiny_trades():
    targets = pd.Series({"SPY": 0.495, "BIL": 0.505})
    orders = et.plan_orders(targets, {"SPY": 82, "BIL": 545}, {"SPY": 600.0, "BIL": 91.0}, 100_000, 0)
    assert orders == []


# ---------------------------------------------------------------- live cycle

def test_first_cycle_moves_stocks_into_etfs(cfg):
    closes, opens = make_prices()
    dp = FakeDP(closes, opens, positions=[{"symbol": "TXN", "qty": "17", "market_value": "4930",
                                           "current_price": "290"}], equity=100_000, cash=95_070)
    r = engine_etf.run_etf_cycle(dp, cfg, execute=True, now=NOW)
    assert r["status"] == "executed" and dp.closed == ["TXN"]
    bought = {o["symbol"] for o in dp.orders if o["side"] == "buy"}
    assert "SPY" in bought and "EEM" not in bought and all(o["tif"] == "day" for o in dp.orders)
    st = engine_etf._read_json(engine_etf.STATE_PATH, {})
    assert st["initialized"] and st["last_executed_decision"] == "2026-10-02"


def test_second_cycle_same_day_does_not_trade_again(cfg):
    closes, opens = make_prices()
    dp = FakeDP(closes, opens)
    engine_etf.run_etf_cycle(dp, cfg, execute=True, now=NOW)
    n = len(dp.orders)
    assert n > 0
    engine_etf.run_etf_cycle(dp, cfg, execute=True, now=NOW)
    assert len(dp.orders) == n


def test_before_open_only_plans(cfg):
    closes, opens = make_prices()
    dp = FakeDP(closes, opens, market_open=False)
    r = engine_etf.run_etf_cycle(dp, cfg, execute=True, now=NOW)
    assert dp.orders == [] and r["status"].startswith("planned")
    st = engine_etf._read_json(engine_etf.STATE_PATH, {})
    assert st["pending_plan"] and not st.get("initialized")


def test_monitor_mode_never_orders(cfg):
    closes, opens = make_prices()
    dp = FakeDP(closes, opens)
    engine_etf.run_etf_cycle(dp, cfg, execute=False, now=NOW)
    assert dp.orders == [] and dp.closed == []


def test_working_orders_block_new_orders(cfg):
    closes, opens = make_prices()
    dp = FakeDP(closes, opens, open_orders=[{"symbol": "SPY", "side": "buy"}])
    r = engine_etf.run_etf_cycle(dp, cfg, execute=True, now=NOW)
    assert dp.orders == [] and r["status"].startswith("waiting")


def test_old_stock_stop_order_does_not_block_the_switch(cfg):
    """TXN still has its GTC stop from the stock bot: the switch must go ahead
    (close_position cancels the stop) instead of waiting on it forever."""
    closes, opens = make_prices()
    dp = FakeDP(closes, opens, positions=[{"symbol": "TXN", "qty": "17", "market_value": "4930",
                                           "current_price": "290"}], equity=100_000, cash=95_070,
                open_orders=[{"symbol": "TXN", "side": "sell", "type": "stop", "status": "new"}])
    r = engine_etf.run_etf_cycle(dp, cfg, execute=True, now=NOW)
    assert r["status"] == "executed" and dp.closed == ["TXN"]
    assert [o for o in dp.orders if o["side"] == "buy"]


def test_no_trade_between_tranche_days(cfg):
    closes, opens = make_prices()
    engine_etf._write_json(engine_etf.STATE_PATH, {"initialized": True, "last_executed_decision": "2026-10-01"})
    dp = FakeDP(closes, opens)                                # Oct 2 = 2nd trading day: not a tranche day
    r = engine_etf.run_etf_cycle(dp, cfg, execute=True, now=NOW)
    assert not r["due"] and dp.orders == []


def test_tranche_day_triggers_rebalance(cfg):
    closes, opens = make_prices(end="2026-10-01")             # Oct 1 = 1st trading day: tranche day
    engine_etf._write_json(engine_etf.STATE_PATH, {"initialized": True, "last_executed_decision": "2026-09-23"})
    dp = FakeDP(closes, opens)
    r = engine_etf.run_etf_cycle(dp, cfg, execute=True, now=datetime(2026, 10, 2, 10, 0, tzinfo=ET))
    assert r["due"] and dp.orders


def test_tranche_day_executes_only_once(cfg):
    closes, opens = make_prices(end="2026-10-01")
    engine_etf._write_json(engine_etf.STATE_PATH, {"initialized": True, "last_executed_decision": "2026-09-23"})
    dp = FakeDP(closes, opens)                                # positions never update: a re-run WOULD re-trade
    now = datetime(2026, 10, 2, 10, 0, tzinfo=ET)
    engine_etf.run_etf_cycle(dp, cfg, execute=True, now=now)
    n = len(dp.orders)
    assert n > 0
    for _ in range(3):
        engine_etf.run_etf_cycle(dp, cfg, execute=True, now=now)
    assert len(dp.orders) == n


def test_todays_unfinished_bar_is_ignored(cfg):
    closes, opens = make_prices(end="2026-10-05")
    c, _ = engine_etf.load_prices(FakeDP(closes, opens), SYMS + ["BIL"], now=NOW)
    assert c.index[-1] == pd.Timestamp("2026-10-02")
    c2, _ = engine_etf.load_prices(FakeDP(closes, opens), SYMS + ["BIL"],
                                   now=datetime(2026, 10, 5, 16, 30, tzinfo=ET))
    assert c2.index[-1] == pd.Timestamp("2026-10-05")


def test_prices_downloaded_once_per_day(cfg):
    closes, opens = make_prices()
    dp = FakeDP(closes, opens)
    calls = []
    orig = dp.fetch_yf_history
    dp.fetch_yf_history = lambda s, **k: calls.append(s) or orig(s)
    syms = SYMS + ["BIL"]
    a, _ = engine_etf.load_prices(dp, syms, now=NOW)
    b, _ = engine_etf.load_prices(dp, syms, now=NOW)
    assert len(calls) == len(syms)
    pd.testing.assert_frame_equal(a, b)
    engine_etf.load_prices(dp, syms, now=datetime(2026, 10, 5, 16, 30, tzinfo=ET))   # after close: refresh
    assert len(calls) == 2 * len(syms)


def test_drawdown_halt_allows_only_sells(cfg):
    closes, opens = make_prices()
    with open(engine_etf.EQUITY_CURVE_PATH, "w") as f:
        f.write("date,equity\n2026-09-01,180000\n")   # 44% below peak: past the 40% halt
    dp = FakeDP(closes, opens, positions=[{"symbol": "TXN", "qty": "17", "market_value": "4930",
                                           "current_price": "290"}], equity=100_000, cash=95_070)
    engine_etf.run_etf_cycle(dp, cfg, execute=True, now=NOW)
    assert dp.closed == ["TXN"] and not [o for o in dp.orders if o["side"] == "buy"]


def test_jev_shadow_never_blocks_orders(cfg, monkeypatch):
    from schemas import JevDecision, ModelResult
    cfg["jev"]["enabled"] = True
    calls = []

    class HoldJev:
        def __init__(self, config):
            pass

        def decide(self, snapshot, advisor_result=None, **kw):
            calls.append(snapshot)
            return ModelResult(success=True, data=JevDecision(action="hold", qty=0, stop_price=None,
                                                              confidence=0.9, rationale="no"))
    import jev
    import snapshot
    monkeypatch.setattr(jev, "JevAgent", HoldJev)
    monkeypatch.setattr(snapshot, "build_snapshot", lambda *a, **k: {"snapshot_id": "s"})
    tags = []
    monkeypatch.setattr(engine, "_record_signal_forecast", lambda *a: tags.append(a[-1]) or "fc")
    closes, opens = make_prices()
    dp = FakeDP(closes, opens)
    engine_etf.run_etf_cycle(dp, cfg, execute=True, now=NOW)
    assert [o for o in dp.orders if o["side"] == "buy"]          # orders went through
    assert calls and len(calls) <= 3 and set(tags) == {"vetoed"}   # Jev's "hold" logged as a would-be veto


# ---------------------------------------------------------------- replay / monitor / readiness

def test_replay_tracks_the_model():
    closes, opens = make_prices(n=900, seed=3)
    with open(os.path.join(BOT_DIR, "config.yaml")) as f:
        config = yaml.safe_load(f)
    r = engine_etf.replay(closes, opens, closes.index[400], config)
    assert r["orders"] > 0
    assert r["tracking_error_pct"] < 2.0
    assert abs(r["replay_return_pct"] - r["model_return_pct"]) < 3.0


def test_monitor_compare_and_readiness(cfg):
    from learning.etf_monitor import compare
    from learning.readiness import assess_etf
    closes, opens = make_prices(n=700, seed=5)
    params = et.params_from_config(cfg)
    model = et.model_equity(closes, opens, closes.index[-300], params)
    paper = model * 100_000
    c = compare(paper, closes, opens, params)
    assert c["tracking_error_pct"] < 0.01                    # identical → no tracking error
    rd = assess_etf(cfg, paper, c)
    assert rd["mode"] == "etf_trend" and rd["checks"]["tracking_error"]
    assert rd["checks"]["min_days"] and rd["status"] in ("READY_FOR_REVIEW", "NOT_READY")
    short = assess_etf(cfg, paper.iloc[:20], compare(paper.iloc[:20], closes, opens, params))
    assert not short["checks"]["min_days"] and short["status"] == "NOT_READY"


@pytest.mark.parametrize("case", ["missing", "short", "long_gap"])
def test_incomplete_price_history_blocks_held_etf_without_orders(cfg, case):
    closes, opens = make_prices(end="2026-10-01")
    if case == "missing":
        closes, opens = closes.drop(columns="SPY"), opens.drop(columns="SPY")
    elif case == "short":
        closes.loc[closes.index[:-150], "SPY"] = np.nan
    else:
        closes.loc[closes.index[-50:-47], "SPY"] = np.nan
    dp = FakeDP(closes, opens, positions=[{"symbol": "SPY", "qty": "10", "current_price": "100",
                                           "market_value": "1000"}])
    engine_etf._write_json(engine_etf.STATE_PATH, {"initialized": True, "sentinel": "keep"})
    r = engine_etf.run_etf_cycle(dp, cfg, execute=True, now=datetime(2026, 10, 2, 10, tzinfo=ET))
    st = engine_etf._read_json(engine_etf.STATE_PATH, {})
    assert r == {"error": "incomplete_data", "symbols": ["SPY"]}
    assert st["status"].startswith("blocked — incomplete price data (SPY)")
    assert st["sentinel"] == "keep" and dp.orders == [] and dp.closed == []


def test_single_interior_gap_fills_closes_and_opens_and_preserves_targets(cfg):
    clean, opens = make_prices(end="2026-10-01")
    hole = clean.index[-40]
    # Match the intended ffill value exactly to isolate the effect of missing data.
    clean.loc[hole, "SPY"] = clean.loc[clean.index[clean.index.get_loc(hole) - 1], "SPY"]
    opens.loc[hole, "SPY"] = opens.loc[opens.index[opens.index.get_loc(hole) - 1], "SPY"]
    damaged_c, damaged_o = clean.copy(), opens.copy()
    damaged_c.loc[hole, "SPY"] = np.nan
    damaged_o.loc[hole, "SPY"] = np.nan
    now = datetime(2026, 10, 2, 10, tzinfo=ET)
    expected_c, _ = engine_etf.load_prices(FakeDP(clean, opens), SYMS + ["BIL"], now, use_cache=False)
    actual_c, actual_o = engine_etf.load_prices(FakeDP(damaged_c, damaged_o), SYMS + ["BIL"], now, use_cache=False)
    assert actual_c.at[hole, "SPY"] == expected_c.at[hole, "SPY"]
    assert actual_o.at[hole, "SPY"] == opens.at[hole, "SPY"]
    a, _ = et.targets_asof(expected_c)
    b, _ = et.targets_asof(actual_c)
    pd.testing.assert_series_equal(a, b, atol=1e-9)
    dp = FakeDP(damaged_c, damaged_o)
    r = engine_etf.run_etf_cycle(dp, cfg, execute=True, now=now)
    assert r["status"] == "executed" and dp.orders


def test_missing_latest_bar_keeps_existing_trailing_drop(cfg):
    closes, opens = make_prices(end="2026-10-01")
    closes.loc[closes.index[-1], "SPY"] = np.nan
    c, _ = engine_etf.load_prices(FakeDP(closes, opens), SYMS + ["BIL"],
                                  now=datetime(2026, 10, 2, 10, tzinfo=ET), use_cache=False)
    assert c.index[-1] == closes.index[-2] and not c.iloc[-1].isna().any()


def test_reconcile_fills_buy_sell_deduplicates_and_retries_errors(cfg, tmp_path, monkeypatch):
    from learning import etf_fills
    trade = tmp_path / "trade_log.csv"
    fills = tmp_path / "etf_fills.csv"
    monkeypatch.setattr(etf_fills, "TRADE_PATH", str(trade))
    monkeypatch.setattr(etf_fills, "FILL_PATH", str(fills))
    with trade.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "order_id", "symbol", "action", "qty", "price"])
        w.writeheader()
        w.writerows([{"source": "etf_trend", "order_id": oid, "symbol": sym,
                      "action": side, "qty": "10", "price": "100"}
                     for oid, sym, side in [("b", "SPY", "buy"), ("s", "BIL", "sell"), ("pending", "EEM", "buy")]])
    class Broker:
        def get_order_by_id(self, oid):
            return SimpleNamespace(status="filled" if oid != "pending" else "new", filled_avg_price="101",
                                   filled_qty="10", submitted_at="2026-10-06T13:30:00Z",
                                   filled_at="2026-10-06T13:30:01Z")
    dp = SimpleNamespace(trading_client=Broker())
    etf_fills.reconcile_fills(dp)
    etf_fills.reconcile_fills(dp)
    with fills.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {r["order_id"]: float(r["slippage_bp"]) for r in rows} == {"b": 100., "s": -100.}
    fills.unlink()
    dp.trading_client.get_order_by_id = lambda oid: (_ for _ in ()).throw(RuntimeError("broker down"))
    etf_fills.reconcile_fills(dp)
    assert not fills.exists()


def test_etf_report_symbol_list_includes_benchmarks(cfg, monkeypatch, tmp_path):
    from learning import etf_monitor
    closes, opens = make_prices(end="2026-10-01")
    seen = []
    def fake_load(dp, symbols, now):
        seen.extend(symbols)
        raise RuntimeError("reached price loader")
    monkeypatch.setattr(engine_etf, "load_prices", fake_load)
    with pytest.raises(RuntimeError, match="reached price loader"):
        etf_monitor.report(FakeDP(closes, opens), cfg)
    assert set(SYMS + ["BIL"]) <= set(seen)


def test_etf_report_includes_reconciled_fill_slippage(cfg, monkeypatch, tmp_path):
    from learning import etf_monitor, etf_fills, readiness
    closes, opens = make_prices(end="2026-10-01")
    monkeypatch.setattr(engine_etf, "load_prices", lambda dp, symbols, now: (closes, opens))
    monkeypatch.setattr(etf_monitor, "paper_equity", lambda: pd.Series(dtype=float))
    monkeypatch.setattr(readiness, "assess_etf_and_notify", lambda config, paper, comparison:
                        {"status": "NOT_READY", "summary": "waiting"})
    monkeypatch.setattr(etf_monitor, "REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(etf_fills, "reconcile_fills", lambda dp: [{"slippage_bp": "-2"}, {"slippage_bp": "6"}])
    path, _, _ = etf_monitor.report(FakeDP(closes, opens), cfg)
    text = open(path).read()
    assert "2 fills" in text and "average 2.00 bp" in text and "worst 6.00 bp" in text
