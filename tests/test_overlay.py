"""Index + leverage overlay (option B): target maths, order planning with margin, live-cycle guards, model/replay."""
import copy

import numpy as np
import pandas as pd
import pytest

import engine_etf
from strategies import etf_trend as et
from strategies import overlay as ov
from tests.test_etf_trend import FakeDP, NOW, cfg, make_prices  # noqa: F401  (cfg is a fixture)

ON = {"enabled": True, "index_symbol": "SPY", "index_weight": 0.4, "leverage": 1.5, "max_gross": 1.6,
      "borrow_rate_pct": 7.0, "require_margin": True}
TREND = pd.Series({"SPY": 0.10, "EFA": 0.10, "IEF": 0.15, "GLD": 0.15, "BIL": 0.50})   # 50% risky


class MarginDP(FakeDP):
    def __init__(self, *a, multiplier="2", **k):
        super().__init__(*a, **k)
        self.multiplier = multiplier

    def get_account(self):
        return {**super().get_account(), "multiplier": self.multiplier}


def _with(config, **over):
    c = copy.deepcopy(config)
    c["etf_trend"]["overlay"] = {**ON, **over}
    return c


# ---------------------------------------------------------------- target maths

def test_disabled_returns_trend_targets_unchanged():
    out = ov.apply(TREND, ov.settings({"overlay": {**ON, "enabled": False}}))
    pd.testing.assert_series_equal(out, TREND)
    assert ov.label({"overlay": {**ON, "enabled": False}}, "v2") == "v2"


def test_mix_and_netting():
    out = ov.apply(TREND, ov.settings({"overlay": ON}))
    # SPY = 1.5 x (0.4 + 0.6 x 0.10); others = 1.5 x 0.6 x w
    assert out["SPY"] == pytest.approx(1.5 * (0.4 + 0.06))
    assert out["IEF"] == pytest.approx(1.5 * 0.6 * 0.15)
    assert ov.gross(out) == pytest.approx(1.5 * (0.4 + 0.6 * 0.5))       # 1.05
    assert out["BIL"] == 0.0                                              # T-bills used before borrowing
    assert ov.borrow_limit(out) == pytest.approx(0.05)


def test_trend_all_cash_needs_no_borrowing():
    out = ov.apply(pd.Series({"BIL": 1.0}), ov.settings({"overlay": ON}))
    assert out["SPY"] == pytest.approx(0.6) and out["BIL"] == pytest.approx(0.4)
    assert ov.borrow_limit(out) == 0.0


def test_exposure_cap_scales_down():
    out = ov.apply(pd.Series({"SPY": 0.5, "EFA": 0.5, "BIL": 0.0}),
                   ov.settings({"overlay": {**ON, "leverage": 2.0, "max_gross": 1.6}}))
    assert ov.gross(out) == pytest.approx(1.6)
    capped = ov.apply(TREND, ov.settings({"overlay": ON}), max_gross=1.0)
    assert ov.gross(capped) == pytest.approx(1.0) and ov.borrow_limit(capped) == 0.0


@pytest.mark.parametrize("bad", [{"leverage": 3.0}, {"leverage": 0.5}, {"index_weight": 1.2},
                                 {"max_gross": 2.5}, {"borrow_rate_pct": 90}])
def test_bad_settings_are_refused(bad):
    with pytest.raises(ValueError):
        ov.settings({"overlay": {**ON, **bad}})


def test_label_marks_the_new_track_record():
    assert ov.label({"overlay": ON}, "etf_trend_v2.0") == "etf_trend_v2.0+spy40x1.5"


# ---------------------------------------------------------------- order planning

def test_plan_orders_uses_margin_only_up_to_borrow_limit():
    t = ov.apply(TREND, ov.settings({"overlay": ON}))
    prices = {"SPY": 600.0, "EFA": 90.0, "IEF": 95.0, "GLD": 300.0, "BIL": 91.5}
    orders = et.plan_orders(t, {}, prices, 100_000, 100_000, borrow_limit=ov.borrow_limit(t))
    spent = sum(o["qty"] * o["price"] for o in orders if o["side"] == "buy")
    assert 1.00 * 100_000 < spent <= 100_000 * (1 + ov.borrow_limit(t))
    no_margin = et.plan_orders(t, {}, prices, 100_000, 100_000)            # default: no borrowing
    assert sum(o["qty"] * o["price"] for o in no_margin if o["side"] == "buy") <= 99_000 + 1e-6


# ---------------------------------------------------------------- live cycle

def test_margin_account_buys_with_leverage(cfg):
    closes, opens = make_prices(up=tuple(et.DEFAULTS["symbols"]))           # everything trending → borrow
    dp = MarginDP(closes, opens, equity=100_000, cash=100_000, multiplier="2")
    r = engine_etf.run_etf_cycle(dp, _with(cfg), execute=True, now=NOW)
    assert r["status"] == "executed"
    px = closes.iloc[-1]
    spent = sum(o["qty"] * px[o["symbol"]] for o in dp.orders if o["side"] == "buy")
    st = engine_etf._read_json(engine_etf.STATE_PATH, {})
    assert st["gross_target"] > 1.0 and spent > 100_000                   # borrowed
    assert spent <= 100_000 * 1.6
    assert st["strategy_version"] == "etf_trend_v2.0+spy40x1.5"


def test_cash_account_is_capped_at_no_borrowing(cfg):
    closes, opens = make_prices()
    dp = MarginDP(closes, opens, equity=100_000, cash=100_000, multiplier="1")
    engine_etf.run_etf_cycle(dp, _with(cfg), execute=True, now=NOW)
    px = closes.iloc[-1]
    spent = sum(o["qty"] * px[o["symbol"]] for o in dp.orders if o["side"] == "buy")
    st = engine_etf._read_json(engine_etf.STATE_PATH, {})
    assert spent <= 100_000 and st["gross_target"] <= 1.0 + 1e-9
    assert "margin not available" in st["overlay_note"]


def test_exposure_guard_blocks_oversized_plan(cfg, monkeypatch):
    closes, opens = make_prices()
    dp = MarginDP(closes, opens, equity=100_000, cash=100_000)
    real = engine_etf.decide

    def oversized(*a, **k):
        d = real(*a, **k)
        d["orders"] = d["orders"] + [{"symbol": "SPY", "side": "buy", "qty": 1000.0,
                                      "price": float(closes["SPY"].iloc[-1]), "reason": "add"}]
        return d
    monkeypatch.setattr(engine_etf, "decide", oversized)
    r = engine_etf.run_etf_cycle(dp, _with(cfg), execute=True, now=NOW)
    assert dp.orders == [] and dp.closed == [] and r["status"].startswith("blocked")


def test_drawdown_halt_pays_down_borrowing(cfg):
    closes, opens = make_prices()
    held = {"SPY": 1200, "EFA": 300}                                       # ~ $1.3x exposure on $100k
    px = closes.iloc[-1]
    positions = [{"symbol": s, "qty": str(q), "market_value": str(q * px[s]), "current_price": str(px[s])}
                 for s, q in held.items()]
    long_ = sum(q * px[s] for s, q in held.items())
    with open(engine_etf.EQUITY_CURVE_PATH, "w") as f:
        f.write("date,equity\n2026-09-01,200000\n")                       # 50% below peak
    dp = MarginDP(closes, opens, positions=positions, equity=100_000, cash=100_000 - long_)
    engine_etf.run_etf_cycle(dp, _with(cfg), execute=True, now=NOW)
    assert not [o for o in dp.orders if o["side"] == "buy"]
    sold = sum(o["qty"] * px[o["symbol"]] for o in dp.orders if o["side"] == "sell") + \
        sum(held[s] * px[s] for s in dp.closed)
    assert long_ - sold <= 100_000 * 1.01                                  # back to no borrowing


def test_overlay_off_is_the_old_bot(cfg):
    closes, opens = make_prices()
    off = _with(cfg, enabled=False)
    a = engine_etf.decide(closes, {}, closes.iloc[-1].to_dict(), 100_000, 100_000, {}, off["etf_trend"],
                          et.params_from_config(off), None)
    trend, _ = et.targets_asof(closes, None, et.params_from_config(off))
    pd.testing.assert_series_equal(a["targets"], trend)
    assert a["label"] == et.VERSION


# ---------------------------------------------------------------- model / replay

def test_model_charges_interest_and_replay_tracks_it(cfg):
    closes, opens = make_prices(n=900, seed=3)
    c = _with(cfg)
    r = engine_etf.replay(closes, opens, closes.index[400], c)
    assert r["orders"] > 0 and r["tracking_error_pct"] < 2.0
    assert abs(r["replay_return_pct"] - r["model_return_pct"]) < 3.0
    p = et.params_from_config(c)
    cheap = et.model_equity(closes, opens, closes.index[400], {**p, "overlay": {**ON, "borrow_rate_pct": 0.0}})
    dear = et.model_equity(closes, opens, closes.index[400], {**p, "overlay": {**ON, "borrow_rate_pct": 15.0}})
    assert dear.iloc[-1] < cheap.iloc[-1]                                  # borrowing is not free


# ---------------------------------------------------------------- monitor

def test_monitor_report_calls_paper_equity_without_arguments(cfg, tmp_path, monkeypatch):
    """Other tests replace paper_equity with a no-argument fake; report() must keep working."""
    import learning.etf_monitor as mon
    import learning.readiness as rd
    closes, opens = make_prices(n=700, seed=5)
    monkeypatch.setattr(mon, "REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setattr(rd, "STATE_PATH", str(tmp_path / "readiness_state.json"))
    model = et.model_equity(closes, opens, closes.index[-60], et.params_from_config(cfg))
    monkeypatch.setattr(mon, "paper_equity", lambda: model * 100_000)
    path, cmp_, res = mon.report(FakeDP(closes, opens), cfg, now=NOW)
    assert cmp_["tracking_error_pct"] < 0.01 and "etf_trend_v2.0+spy40x1.5" in open(path).read()


def test_paper_equity_defaults_to_the_configured_label(tmp_path):
    import learning.etf_monitor as mon
    p = tmp_path / "equity_curve.csv"
    p.write_text("date,equity,strategy_version\n"
                 "2026-10-01,100000,etf_trend_v2.0\n"
                 "2026-10-02,101000,etf_trend_v2.0+spy40x1.5+ai\n"
                 "2026-10-05,102000,etf_trend_v2.0+spy40x1.5+ai\n")
    assert mon.config_label() == "etf_trend_v2.0+spy40x1.5+ai"           # shipped config: overlay + AI on
    assert list(mon.paper_equity(str(p))) == [101000.0, 102000.0]
    assert list(mon.paper_equity(str(p), version="etf_trend_v2.0")) == [100000.0]
