"""End-to-end tests for the trading engine against a fake broker (no network)."""
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine  # noqa: E402
from risk import RiskManager  # noqa: E402
from strategies.v1_1 import SMACrossV1_1  # noqa: E402


def _frame(closes):
    idx = pd.date_range(end=datetime(2026, 10, 2, tzinfo=timezone.utc), periods=len(closes), freq="B")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e6})


def fresh_cross_up():
    """Downtrend then rally, truncated on the bar SMA20 crosses above SMA50."""
    closes = list(np.linspace(150, 100, 100)) + list(np.linspace(100, 160, 60))
    df = _frame(closes)
    s = df["Close"].rolling(20).mean() > df["Close"].rolling(50).mean()
    first = int(np.argmax(s.values & ~s.shift(1, fill_value=False).values & (np.arange(len(s)) > 60)))
    return df.iloc[: first + 1]


def downtrend():
    return _frame(list(np.linspace(100, 160, 80)) + list(np.linspace(160, 110, 70)))


class FakeDP:
    """Minimal broker double recording orders."""
    paper = True
    data_client = None

    def __init__(self, bars, positions=None, equity=100_000.0, last_equity=100_000.0, market_open=True):
        self.trading_client = object()
        self.bars = bars
        self.positions = positions or []
        self.equity, self.last_equity, self.market_open = equity, last_equity, market_open
        self.orders, self.closed, self.replaced = [], [], []

    def get_account(self):
        return {"equity": str(self.equity), "last_equity": str(self.last_equity),
                "cash": str(self.equity), "portfolio_value": str(self.equity), "currency": "USD"}

    def get_positions(self):
        return self.positions

    def get_open_orders(self, symbol=None):
        return []

    def get_clock(self):
        return {"is_open": self.market_open}

    def fetch_yf_history(self, symbol, period="1y", interval="1d"):
        return self.bars.get(symbol, pd.DataFrame())

    def fetch_yf_fundamentals(self, symbol):
        return {}

    def place_order(self, symbol, qty, side="buy", time_in_force="day", stop_loss_price=None):
        self.orders.append({"symbol": symbol, "qty": qty, "side": side, "stop": stop_loss_price})
        return {"id": f"ord-{len(self.orders)}"}

    def close_position(self, symbol):
        self.closed.append(symbol)
        return {"id": f"close-{symbol}"}

    def replace_stop(self, order_id, stop_price):
        self.replaced.append((order_id, stop_price))
        return {}


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    for name in ("LOCK_PATH", "RISK_STATE_PATH", "POSITIONS_STATE_PATH", "ADVISOR_STATE_PATH",
                 "CLOSED_TRADES_PATH", "UNIVERSE_CACHE_PATH", "ACTIVE_STRATEGY_PATH",
                 "DECISION_POLICY_PATH", "EQUITY_CURVE_PATH", "SHADOW_CACHE_DIR"):
        monkeypatch.setattr(engine, name, str(tmp_path / name.lower()))
    monkeypatch.setenv("FINBOT_NOTIFY", "off")
    import journal.trade_log as tl
    import journal.audit_log as al
    import research.roles as roles
    monkeypatch.setattr(tl, "LOG_PATH", str(tmp_path / "trade_log.csv"))
    monkeypatch.setattr(al, "LOG_PATH", str(tmp_path / "audit_log.csv"))
    monkeypatch.setattr(roles, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(engine, "_record_signal_forecast", lambda *a, **k: "fc-test")
    return {
        "market_universe": {"mode": "static", "symbols": ["UP", "DN"]},
        "data": {"lookback_days": 150},
        "execution": {"auto_trade": True, "require_market_open": True},
        "advisor": {"enabled": True, "provider": "mock", "interval_minutes": 0, "max_calls_per_cycle": 1},
        "jev": {"enabled": True, "provider": "mock", "min_confidence_entry": 0.55,
                "min_confidence_exit": 0.6, "review_open_positions": True},
        "strategy": {"version": "v1.1", "sma_short": 20, "sma_long": 50, "entry_lookback_bars": 3},
        "risk_limits": {"per_trade_max_loss_usd": 2000.0, "total_open_exposure_usd": 10000.0,
                        "max_position_size_usd": 5000.0, "daily_loss_limit_usd": 5000.0,
                        "drawdown_halt_pct": 15.0, "min_order_collateral_usd": 15.0},
        "paper_trading": {"base_balance_usd": 100000.0},
        "research": {"enabled": True, "roles": ["technical", "bull_bear"]},
        "journal": {"log_no_signal": False},
        "dashboard": {"enabled": False},
    }


def test_live_signal_matches_crossover():
    st = SMACrossV1_1()
    assert st.live_signal(fresh_cross_up()) == 1
    assert st.live_signal(downtrend()) == -1


def test_entry_places_oto_order_within_limits(cfg):
    dp = FakeDP({"UP": fresh_cross_up(), "DN": downtrend()})
    counts = engine.run_cycle(dp, cfg, execute=True)
    assert counts["orders"] == 1
    o = dp.orders[0]
    close = float(fresh_cross_up()["Close"].iloc[-1])
    assert o["symbol"] == "UP" and o["side"] == "buy"
    assert 0 < o["stop"] < close
    assert o["qty"] * close <= 5000.0 + 1e-6           # max position
    assert o["qty"] * (close - o["stop"]) <= 2000.0     # per-trade risk
    assert counts["no_signal"] == 1                     # DN never reached the models


def test_monitor_mode_never_orders(cfg):
    dp = FakeDP({"UP": fresh_cross_up(), "DN": downtrend()})
    engine.run_cycle(dp, cfg, execute=False)
    assert dp.orders == [] and dp.closed == []


def test_market_closed_blocks_orders(cfg):
    dp = FakeDP({"UP": fresh_cross_up(), "DN": downtrend()}, market_open=False)
    engine.run_cycle(dp, cfg, execute=True)
    assert dp.orders == []


def test_held_position_exits_on_rule_even_when_halted(cfg):
    df = downtrend()
    pos = [{"symbol": "HLD", "qty": "10", "avg_entry_price": "150", "market_value": str(10 * df["Close"].iloc[-1])}]
    # equity 20% below persisted peak → drawdown halt for NEW entries
    dp = FakeDP({"UP": fresh_cross_up(), "DN": downtrend(), "HLD": df}, positions=pos, equity=80_000)
    RiskManager(cfg, 100_000).save_state(engine.RISK_STATE_PATH)
    engine.run_cycle(dp, cfg, execute=True)
    assert dp.closed == ["HLD"]                # exit allowed while halted
    assert dp.orders == []                     # no new entries while halted
    trades = engine.read_closed_trades(engine.CLOSED_TRADES_PATH)
    assert trades and trades[0]["symbol"] == "HLD" and trades[0]["exit_reason"] == "sma_cross_exit_signal"


def test_exposure_cap_counts_existing_positions(cfg):
    # Already at the $10k exposure cap → no new order, no model calls needed
    pos = [{"symbol": "AAA", "qty": "100", "avg_entry_price": "100", "market_value": "10000"}]
    dp = FakeDP({"UP": fresh_cross_up(), "DN": downtrend()}, positions=pos)
    engine.run_cycle(dp, cfg, execute=True)
    assert dp.orders == []


def test_overlapping_cycle_is_skipped(cfg):
    dp = FakeDP({"UP": fresh_cross_up(), "DN": downtrend()})
    with engine.cycle_lock(engine.LOCK_PATH):
        assert engine.run_cycle(dp, cfg, execute=True) == {"skipped": "locked"}
    assert dp.orders == []


def test_advisor_budget_is_per_call(tmp_path):
    b = engine.AdvisorBudget({"advisor": {"interval_minutes": 15, "max_calls_per_cycle": 2}},
                             enabled=True, state_file=str(tmp_path / "adv"))
    assert [b.take() for _ in range(5)] == [True, True, False, False, False]
    # interval now running → a new cycle gets no calls
    b2 = engine.AdvisorBudget({"advisor": {"interval_minutes": 15, "max_calls_per_cycle": 2}},
                              enabled=True, state_file=str(tmp_path / "adv"))
    assert b2.take() is False


def test_place_order_builds_request_object():
    import data as data_mod
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderClass

    captured = {}

    class TC:
        def submit_order(self, order_data):
            captured["req"] = order_data
            return {"id": "x"}

    dp = data_mod.DataProvider.__new__(data_mod.DataProvider)
    dp.trading_client, dp.paper = TC(), True
    dp.place_order("AAPL", 3.7, "buy", stop_loss_price=180.123)
    req = captured["req"]
    assert isinstance(req, MarketOrderRequest)
    assert req.order_class == OrderClass.OTO and req.qty == 3 and req.stop_loss.stop_price == 180.12


def test_place_order_refuses_live():
    import data as data_mod
    dp = data_mod.DataProvider.__new__(data_mod.DataProvider)
    dp.trading_client, dp.paper = object(), False
    with pytest.raises(RuntimeError):
        dp.place_order("AAPL", 1, "buy")


def test_dynamic_universe_screens_and_ranks(cfg):
    class DP:
        def get_all_assets_meta(self):
            mk = lambda s, n="Common Stock": {"symbol": s, "asset_class": "us_equity", "tradable": True,
                                              "status": "active", "exchange": "NASDAQ", "name": n}
            return [mk("BIG"), mk("MID"), mk("PENNY"), mk("ABC.WS"), mk("ADSEW", "Foo Corp Warrants"),
                    mk("ILLIQ")]

        def get_snapshots(self, syms):
            data = {"BIG": (100, 5e6), "MID": (50, 1e6), "PENNY": (0.5, 1e9), "ILLIQ": (20, 1000),
                    "ADSEW": (1, 1e6)}
            return {s: {"price": p, "volume": v, "prev_volume": v, "prev_close": p}
                    for s, (p, v) in data.items() if s in syms}

    cfg["market_universe"] = {"mode": "dynamic", "dynamic": {"max_symbols": 10, "min_price": 5,
                                                              "min_dollar_volume": 1e7}}
    assert engine.resolve_market_universe(cfg, DP()) == ["BIG", "MID"]


def test_risk_sync_from_broker():
    rm = RiskManager({"risk_limits": {"daily_loss_limit_usd": 1000}}, 100_000)
    rm.sync_from_broker({"equity": "98000", "last_equity": "99500"},
                        [{"symbol": "X", "qty": "10", "market_value": "1500"}])
    assert rm.total_open_exposure() == 1500 and rm.daily_summary()["loss"] == 1500
    chk = rm.check_trade("Y", "buy", 10, 50, 49)
    assert not chk.passed and rm.halted          # daily loss limit already breached
    assert rm.check_trade("X", "sell", 10, 150, None, held_qty=10).passed


def test_scheduled_gate_is_dst_safe(monkeypatch):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import run_scheduled as rs
    from datetime import time as t
    for utc, expected in [("2026-10-05T12:50:00+00:00", True),   # 05:50 PDT
                          ("2026-12-07T13:50:00+00:00", True),   # 05:50 PST
                          ("2026-12-07T13:40:00+00:00", False),  # 05:40 PST
                          ("2026-10-05T20:05:00+00:00", False),  # 13:05 PDT
                          ("2026-10-04T17:00:00+00:00", False)]: # Sunday
        monkeypatch.setenv("FINBOT_FAKE_NOW", utc)
        assert rs.in_window(rs.now_pacific(), t(5, 45), t(13, 0)) is expected, utc


def test_learned_policy_can_override_jev_veto(cfg):
    import json
    with open(engine.DECISION_POLICY_PATH, "w") as f:
        json.dump({"min_confidence_entry": 0.95, "jev_can_veto": False}, f)
    dp = FakeDP({"UP": fresh_cross_up(), "DN": downtrend()})
    engine.run_cycle(dp, cfg, execute=True)
    assert len(dp.orders) == 1          # Jev conf < 0.95 but veto disabled by policy


def test_learned_threshold_blocks_low_confidence(cfg):
    import json
    with open(engine.DECISION_POLICY_PATH, "w") as f:
        json.dump({"min_confidence_entry": 0.95, "jev_can_veto": True}, f)
    dp = FakeDP({"UP": fresh_cross_up(), "DN": downtrend()})
    engine.run_cycle(dp, cfg, execute=True)
    assert dp.orders == []


class FlakyJev:
    """Stochastic Jev double: returns the queued confidences in order."""
    provider, model = "flaky", "test"

    def __init__(self, confs, action="buy"):
        self.confs, self.action, self.calls = list(confs), action, 0

    def __call__(self, config):
        return self

    def decide(self, snap, adv, findings=None):
        from schemas import JevDecision, ModelResult
        self.calls += 1
        close = snap["market"]["close"]
        return ModelResult(success=True, data=JevDecision(
            action=self.action if self.action in snap["valid_actions"] else "hold",
            qty=0, stop_price=round(close * 0.97, 2), rationale="flaky",
            confidence=self.confs.pop(0)))


def test_jev_veto_is_not_re_asked_same_day(cfg, monkeypatch):
    jev = FlakyJev([0.12, 0.67])          # the 2026-10-05 TXN sequence, compressed
    monkeypatch.setattr(engine, "JevAgent", jev)
    dp = FakeDP({"UP": fresh_cross_up(), "DN": downtrend()})
    engine.run_cycle(dp, cfg, execute=True)
    engine.run_cycle(dp, cfg, execute=True)
    assert jev.calls == 1 and dp.orders == []   # veto stands; no retry-until-yes


def test_approved_entry_retried_without_re_asking(cfg, monkeypatch):
    jev = FlakyJev([0.9])
    monkeypatch.setattr(engine, "JevAgent", jev)
    closed = FakeDP({"UP": fresh_cross_up(), "DN": downtrend()}, market_open=False)
    engine.run_cycle(closed, cfg, execute=True)
    assert closed.orders == []
    opened = FakeDP({"UP": fresh_cross_up(), "DN": downtrend()})
    engine.run_cycle(opened, cfg, execute=True)
    assert jev.calls == 1 and len(opened.orders) == 1


def test_held_position_reviewed_once_per_day(cfg, monkeypatch):
    jev = FlakyJev([0.3, 0.9], action="sell")
    monkeypatch.setattr(engine, "JevAgent", jev)
    df = fresh_cross_up()
    pos = [{"symbol": "UP", "qty": "10", "avg_entry_price": str(df["Close"].iloc[-1]),
            "market_value": str(10 * df["Close"].iloc[-1])}]
    dp = FakeDP({"UP": df, "DN": downtrend()}, positions=pos)
    engine.run_cycle(dp, cfg, execute=True)
    engine.run_cycle(dp, cfg, execute=True)
    assert jev.calls == 1 and dp.closed == []   # a later lucky 0.9 sell never gets rolled


def test_entry_uses_strategy_atr_stop_not_jev_fixed_pct(cfg):
    df = fresh_cross_up()
    dp = FakeDP({"UP": df, "DN": downtrend()})
    engine.run_cycle(dp, cfg, execute=True)
    expected = SMACrossV1_1().initial_stop(df)
    assert dp.orders[0]["stop"] == pytest.approx(expected)
    assert dp.orders[0]["stop"] != pytest.approx(float(df["Close"].iloc[-1]) * 0.97)


def test_alpaca_bars_are_split_adjusted():
    import data as data_mod
    from alpaca.data.enums import Adjustment
    seen = []

    class DC:
        def get_stock_bars(self, req):
            seen.append(req)
            return type("R", (), {"df": pd.DataFrame()})()

    dp = data_mod.DataProvider.__new__(data_mod.DataProvider)
    dp.data_client = DC()
    now = datetime.now(timezone.utc)
    dp.fetch_alpaca_bars_multi(["NVDA"], now - timedelta(days=10), now)
    dp.fetch_alpaca_bars("NVDA", now - timedelta(days=10), now)
    assert len(seen) == 2 and all(r.adjustment == Adjustment.ALL for r in seen)
