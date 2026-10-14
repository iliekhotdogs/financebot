"""AI portfolio manager (ai_pm.py): limits, one call a day, fallbacks, yes/no questions, live-cycle wiring."""
import copy
import json
from datetime import datetime

import pandas as pd
import pytest

import ai_pm
import engine_etf
from strategies import etf_trend as et
from tests.test_etf_trend import ET, FakeDP, NOW, cfg, make_prices  # noqa: F401  (cfg is a fixture)

SYMS = et.DEFAULTS["symbols"]
S = ai_pm.settings({"etf_trend": {"ai": {"enabled": True}}})
RULES = pd.Series({"SPY": 0.70, "IEF": 0.30, "GLD": 0.30, "TLT": 0.20, "BIL": 0.0})   # gross 1.5
TUE = datetime(2026, 10, 6, 10, 0, tzinfo=ET)   # last completed session Mon Oct 5 (not a rebalance day)


class MarginDP(FakeDP):
    def get_account(self):
        return {**super().get_account(), "multiplier": "2"}


class Model:
    """Fake model: returns the queued answers in order and counts calls."""
    def __init__(self, *answers):
        self.answers, self.calls, self.briefings = list(answers), 0, []

    def __call__(self, s, briefing):
        self.calls += 1
        self.briefings.append(briefing)
        a = self.answers.pop(0) if self.answers else {"tilts": {}, "hold": [], "summary": "no change"}
        return (None, a) if isinstance(a, str) else (a, None)


def ai_cfg(config, **over):
    c = copy.deepcopy(config)
    c["etf_trend"]["ai"] = {**c["etf_trend"].get("ai", {}), "enabled": True, "news_headlines": 0, **over}
    return c


@pytest.fixture
def store(tmp_path, monkeypatch):
    p = str(tmp_path / "ai_pm.json")
    monkeypatch.setattr(ai_pm, "STORE_PATH", p)
    return p


def _plan(config, model, store, asof="2026-10-02", now=NOW, current=None, execute=True, rules=RULES, cap=1.6):
    closes, _ = make_prices()
    _, info = et.targets_asof(closes, None, et.params_from_config(config))
    return ai_pm.daily_plan(config, asof=asof, rules=rules, info=info, closes=closes,
                            current_w=current or dict(rules), equity=100_000, cash=0, peak=100_000, cap=cap,
                            symbols=SYMS, execute=execute, now=now, store_path=store, caller=model)


# ---------------------------------------------------------------- validation and maths

def test_validate_accepts_percent_points():
    plan, err = ai_pm.validate({"tilts": {"spy": 5, "TLT": -3}, "hold": ["EEM"], "confidence": 0.7,
                                "summary": "x"}, SYMS, S)
    assert err is None and plan["tilts"] == {"SPY": 0.05, "TLT": -0.03} and plan["hold"] == ["EEM"]


@pytest.mark.parametrize("raw,why", [
    ({"tilts": {"QQQ": 5}}, "unknown"),
    ({"tilts": {"BIL": 5}}, "unknown"),                       # T-bills absorb the remainder, not tiltable
    ({"tilts": {"SPY": 16}}, "outside"),
    ({"tilts": {"SPY": 15, "IEF": 15, "GLD": 15}}, "total"),
    ({"tilts": {"SPY": 5}, "hold": ["SPY"]}, "both"),
    ({"tilts": {"SPY": "lots"}}, "not a number"),
    ({"tilts": [1, 2]}, "must be"),
    ("not json", "not a JSON object"),
])
def test_validate_rejects_out_of_limit_answers(raw, why):
    plan, err = ai_pm.validate(raw, SYMS, S)
    assert plan is None and why in err


def test_apply_tilts_holds_floor_and_cap():
    out = ai_pm.apply(RULES, {"tilts": {"SPY": 0.10, "TLT": -0.30}, "hold": ["GLD"]}, {"GLD": 0.25}, cap=1.6)
    assert out["SPY"] == pytest.approx(0.80) and "TLT" not in out          # never below zero
    assert out["GLD"] == pytest.approx(0.25)                              # held at its current weight
    assert ai_pm.gross(out) == pytest.approx(1.35) and out["BIL"] == 0.0
    capped = ai_pm.apply(RULES, {"tilts": {"SPY": 0.15}}, {}, cap=1.6)
    assert ai_pm.gross(capped) == pytest.approx(1.6)                       # scaled down to the cap
    unlev = ai_pm.apply(pd.Series({"SPY": 0.5, "BIL": 0.5}), {"tilts": {"SPY": -0.1}}, {}, cap=1.0)
    assert unlev["BIL"] == pytest.approx(0.6)                              # cash takes the remainder


def test_changed_uses_min_change():
    a = {"tilts": {"SPY": 0.05}, "hold": []}
    assert not ai_pm.changed(a, {"tilts": {"SPY": 0.04}}, 0.02)
    assert ai_pm.changed(a, {"tilts": {"SPY": 0.03}}, 0.02)
    assert ai_pm.changed({"tilts": {}, "hold": ["EEM"]}, {}, 0.02)
    assert ai_pm.changed({}, {"tilts": {"SPY": 0.05}}, 0.02)               # dropping a tilt trades back


def test_label_and_settings_limits():
    assert ai_pm.label({"etf_trend": {"ai": {"enabled": True}}}, "v2") == "v2+ai"
    assert ai_pm.label({"etf_trend": {"ai": {"enabled": False}}}, "v2") == "v2"
    with pytest.raises(ValueError):
        ai_pm.settings({"etf_trend": {"ai": {"max_tilt": 0.5}}})


# ---------------------------------------------------------------- one call a day, fallbacks

def test_one_model_call_per_data_date(cfg, store):
    c, m = ai_cfg(cfg), Model({"tilts": {"GLD": 5}, "summary": "gold"})
    p1 = _plan(c, m, store)
    p2 = _plan(c, m, store)
    assert m.calls == 1 and p1["status"] == p2["status"] == "accepted" and p2["tilts"] == {"GLD": 0.05}
    _plan(c, m, store, asof="2026-10-05", now=TUE)
    assert m.calls == 2                                                    # next data date: one more call


def test_failure_is_not_retried_and_keeps_the_current_plan(cfg, store):
    c = ai_cfg(cfg)
    _plan(c, Model({"tilts": {"GLD": 5}}), store)                          # Fri: accepted
    m = Model("TimeoutError")
    p = _plan(c, m, store, asof="2026-10-05", now=TUE)
    p_again = _plan(c, m, store, asof="2026-10-05", now=TUE)
    assert m.calls == 1 and p["status"] == "fallback" and p_again["tilts"] == {"GLD": 0.05}


def test_invalid_answer_falls_back(cfg, store):
    p = _plan(ai_cfg(cfg), Model({"tilts": {"SPY": 40}}), store)
    assert p["status"] == "fallback" and p["tilts"] == {} and "outside" in p["note"]


def test_stale_tilts_expire_to_pure_rules(cfg, store):
    c = ai_cfg(cfg, stale_days=2)
    _plan(c, Model({"tilts": {"GLD": 5}}), store, asof="2026-09-28")
    for d in ("2026-09-29", "2026-09-30"):
        assert _plan(c, Model("down"), store, asof=d)["tilts"] == {"GLD": 0.05}
    assert _plan(c, Model("down"), store, asof="2026-10-01")["tilts"] == {}


def test_no_call_before_decide_after_or_in_monitor_mode(cfg, store):
    c, m = ai_cfg(cfg), Model()
    early = datetime(2026, 10, 5, 8, 30, tzinfo=ET)                       # 05:30 Pacific
    assert _plan(c, m, store, now=early)["status"] == "none"
    assert _plan(c, m, store, execute=False)["status"] == "none"
    assert m.calls == 0


def test_briefing_is_compact_and_has_history(cfg, store):
    c, m = ai_cfg(cfg), Model({"tilts": {"GLD": 5}})
    _plan(c, m, store)
    _plan(c, m, store, asof="2026-10-05", now=TUE)
    b = m.briefings[-1]
    assert len(json.dumps(b, separators=(",", ":"))) < 3000               # ~750 tokens
    assert set(b["etf"]) == set(SYMS) | {"BIL"} and b["past"][0]["tilt"] == {"GLD": 5.0}
    assert b["past"][0]["bp"] is not None


# ---------------------------------------------------------------- yes/no questions

LOW = pd.Series({"SPY": 0.5, "IEF": 0.3, "GLD": 0.2, "BIL": 0.0})          # rules at 100% exposure
BIG = {"tilts": {"SPY": 15, "IEF": 15}, "summary": "risk on"}             # +30 points of exposure → ask


def test_big_move_asks_and_yes_applies(cfg, store, monkeypatch):
    c, asked, answer = ai_cfg(cfg), [], {"v": None}
    monkeypatch.setattr(ai_pm, "ask", lambda config, text: asked.append(text) or "m1")
    monkeypatch.setattr(ai_pm, "read_answer", lambda config, mid, approver="": answer["v"])
    p = _plan(c, Model(BIG), store, rules=LOW)
    assert p["status"] == "pending" and p["tilts"] == {} and "❓" in asked[0]
    assert _plan(c, Model(), store, rules=LOW)["status"] == "pending"      # no answer yet, no re-ask
    answer["v"] = "yes"
    p = _plan(c, Model(), store, rules=LOW)
    assert p["status"] == "approved" and p["tilts"]["IEF"] == 0.15 and len(asked) == 1


def test_no_then_cooldown_and_deadline(cfg, store, monkeypatch):
    c, asked = ai_cfg(cfg), []
    monkeypatch.setattr(ai_pm, "ask", lambda config, text: asked.append(text) or "m1")
    monkeypatch.setattr(ai_pm, "read_answer", lambda config, mid, approver="": "no")
    big = BIG
    _plan(c, Model(big), store, rules=LOW)
    assert _plan(c, Model(), store, rules=LOW)["status"] == "declined"
    p = _plan(c, Model(big), store, asof="2026-10-05", now=TUE, rules=LOW)   # within cooldown: not asked
    assert p["status"] == "declined" and len(asked) == 1
    # an unanswered question expires at the deadline → skip
    monkeypatch.setattr(ai_pm, "read_answer", lambda config, mid, approver="": None)
    c2 = ai_cfg(cfg, cooldown_days=0)
    s2 = store + ".2"
    _plan(c2, Model(big), s2, rules=LOW)
    late = datetime(2026, 10, 5, 15, 30, tzinfo=ET)                        # Mon 12:30 Pacific > deadline
    p = _plan(c2, Model(), s2, now=late, rules=LOW)
    assert p["status"] == "declined" and p["note"] == "expired" and p["tilts"] == {}


def test_cannot_ask_skips_big_move(cfg, store):
    p = _plan(ai_cfg(cfg), Model(BIG), store, rules=LOW)                  # notify is off → can't ask
    assert p["status"] == "declined" and p["note"] == "cannot_ask" and p["tilts"] == {}


def test_read_answer_ignores_bots_and_prefers_no(monkeypatch):
    calls = {}
    def fake(method, path, body=None):
        if "%E2%9C%85" in path:   # ✅
            return calls.get("yes", [])
        return calls.get("no", [])
    monkeypatch.setattr(ai_pm, "_discord", fake)
    calls["yes"] = [{"id": "1", "bot": True}]
    assert ai_pm.read_answer({}, "m") is None
    calls["yes"] = [{"id": "1", "bot": True}, {"id": "42"}]
    assert ai_pm.read_answer({}, "m") == "yes"
    assert ai_pm.read_answer({}, "m", approver="7") is None
    calls["no"] = [{"id": "42"}]
    assert ai_pm.read_answer({}, "m") == "no"


# ---------------------------------------------------------------- live cycle

def test_live_cycle_applies_tilt_and_trades_off_cycle_on_change(cfg, store, monkeypatch):
    c = ai_cfg(cfg)
    closes, opens = make_prices(end="2026-10-05")
    m = Model({"tilts": {"GLD": 5}, "summary": "gold"}, {"tilts": {"GLD": 5}}, {"tilts": {"GLD": 5, "IEF": 4}})
    monkeypatch.setattr(ai_pm, "_call_model", m)
    dp = MarginDP(closes, opens)
    r = engine_etf.run_etf_cycle(dp, c, execute=True, now=TUE)            # first allocation, with AI
    st = engine_etf._read_json(engine_etf.STATE_PATH, {})
    assert r["status"] == "executed" and st["strategy_version"].endswith("+ai")
    assert st["targets"]["GLD"] == pytest.approx(st["rules_targets"].get("GLD", 0) + 0.05, abs=1e-4)
    assert st["ai_executed"] == {"tilts": {"GLD": 0.05}, "hold": []}
    n = len(dp.orders)
    engine_etf.run_etf_cycle(dp, c, execute=True, now=TUE)               # same day: no call, no trade
    assert m.calls == 1 and len(dp.orders) == n
    # next day, not a rebalance day: same tilt → no trade; new tilt → trades the change
    closes2, opens2 = make_prices(end="2026-10-06")
    dp2 = MarginDP(closes2, opens2)
    wed = datetime(2026, 10, 7, 10, 0, tzinfo=ET)
    r2 = engine_etf.run_etf_cycle(dp2, c, execute=True, now=wed)
    assert r2["due"] is False and dp2.orders == []
    closes3, opens3 = make_prices(end="2026-10-07")
    dp3 = MarginDP(closes3, opens3)
    r3 = engine_etf.run_etf_cycle(dp3, c, execute=True, now=datetime(2026, 10, 8, 10, 0, tzinfo=ET))
    assert r3["due"] is True and any(o["symbol"] == "IEF" for o in dp3.orders)


def test_live_cycle_survives_ai_crash(cfg, store, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bug")
    monkeypatch.setattr(ai_pm, "daily_plan", boom)
    closes, opens = make_prices()
    dp = FakeDP(closes, opens)
    r = engine_etf.run_etf_cycle(dp, ai_cfg(cfg), execute=True, now=NOW)
    st = engine_etf._read_json(engine_etf.STATE_PATH, {})
    assert r["status"] == "executed" and st["targets"] == st["rules_targets"]


def test_drawdown_halt_ignores_ai(cfg):
    c = ai_cfg(cfg)
    closes, _ = make_prices()
    params = et.params_from_config(c)
    plain = engine_etf.decide(closes, {}, {}, 50_000, 50_000, {"initialized": True}, c["etf_trend"], params,
                              peak_equity=100_000)
    with_ai = engine_etf.decide(closes, {}, {}, 50_000, 50_000, {"initialized": True}, c["etf_trend"], params,
                                peak_equity=100_000, ai={"tilts": {"SPY": 0.15}, "hold": []}, ai_due=True)
    assert plain["halted"] and with_ai["halted"]
    pd.testing.assert_series_equal(plain["targets"], with_ai["targets"])


def test_readiness_requires_ai_to_beat_rules(cfg):
    from learning.readiness import assess_etf
    c = ai_cfg(cfg)
    comp = {"days": 300, "tracking_error_pct": 4.0, "paper": {"total_return_pct": 9.0, "max_drawdown_pct": -10},
            "model": {"total_return_pct": 10.0}, "spy": {"max_drawdown_pct": -20},
            "sixty_forty": {"max_drawdown_pct": -12}}
    rd = assess_etf(c, pd.Series(dtype=float), comp)
    assert "tracking_error" not in rd["checks"] and rd["checks"]["ai_beats_rules"] is False
    comp["paper"]["total_return_pct"] = 11.0
    assert assess_etf(c, pd.Series(dtype=float), comp)["checks"]["ai_beats_rules"] is True


def test_real_model_is_never_called_under_pytest():
    raw, err = ai_pm._call_model(S, {})
    assert raw is None and "pytest" in err
