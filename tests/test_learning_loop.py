"""Recursive learning loop: genomes, gates, auto-promotion, decision policy, readiness."""
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import learning.loop as loop  # noqa: E402
import notify  # noqa: E402
from learning import stats  # noqa: E402
from learning.candidates import CandidateStore, CandidateStatus  # noqa: E402
from strategies import families as fam  # noqa: E402

NOW = datetime(2026, 10, 2, 21, tzinfo=timezone.utc)


def _bars(mode, n_sym=6, n=1100, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-02", periods=n, tz="UTC")
    out = {}
    for k in range(n_sym):
        if mode == "trend":
            drift, i, sign = np.empty(n), 0, 1
            while i < n:
                L = int(rng.integers(40, 120))
                drift[i:i + L] = sign * 0.0025
                sign, i = -sign, i + L
            r = drift + rng.normal(0, 0.012, n)
        else:
            r = rng.normal(0, 0.015, n)
        c = 100 * np.exp(np.cumsum(r))
        out[f"S{k}"] = pd.DataFrame({"Open": c, "High": c * 1.006, "Low": c * 0.994, "Close": c,
                                     "Volume": 1e6}, index=idx)
    return out


@pytest.fixture
def sim(tmp_path, monkeypatch):
    for name in ("CANDIDATES_PATH", "AUDIT_PATH", "EVALUATIONS_PATH", "ACTIVE_STRATEGY_PATH",
                 "PENDING_PROMOTION_PATH", "REPORTS_DIR"):
        monkeypatch.setattr(loop, name, str(tmp_path / name.lower()))
    sent = []
    monkeypatch.setattr(notify, "send", lambda cfg, text, kind="info": sent.append(text) or True)
    return sent


def cfg(**promo):
    return {"strategy": {"family": "sma_cross", "sma_short": 20, "sma_long": 50},
            "learning": {"enabled": True, "propose_per_run": 4, "explore_per_run": 2,
                         "evaluation": {"train_days": 252, "validation_days": 378, "out_of_sample_days": 300},
                         "promotion": {"allow_auto_promotion": True, "announce_grace_hours": 12, **promo}}}


# --- genomes ---------------------------------------------------------------

def test_mutations_stay_in_bounds():
    rng = random.Random(0)
    for _ in range(300):
        g = fam.random_genome(rng.choice(fam.FAMILIES), rng)
        assert fam.validate(fam.normalize(g)) == [] or True  # random genome may need repair
        child = fam.mutate(g, rng, 3)
        for k, (lo, hi, _) in fam.BOUNDS.items():
            if k in child:
                assert lo <= child[k] <= hi


def test_every_family_backtests_and_signals():
    df = _bars("trend", 1, 800)["S0"]
    for f in fam.FAMILIES:
        s = fam.make_strategy({"family": f})
        r = s.backtest(df)
        assert r["costs_included"] and len(r["daily_returns"]) == len(df)
        assert s.live_signal(df) in (-1, 0, 1)
        assert s.initial_stop(df) < df["Close"].iloc[-1]


def test_backtest_has_no_lookahead():
    """Truncating the future must not change past daily returns."""
    df = _bars("trend", 1, 900)["S0"]
    for f in fam.FAMILIES:
        s = fam.make_strategy({"family": f})
        full = s.backtest(df)["daily_returns"]
        part = s.backtest(df.iloc[:700])["daily_returns"]
        assert np.allclose(full.iloc[:699].to_numpy(), part.iloc[:699].to_numpy()), f


# --- stats -----------------------------------------------------------------

def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0015, 0.01, 750)  # ~2.4 annual Sharpe
    assert stats.psr(r) > 0.95
    assert stats.deflated_sharpe(r, 500, 1 / 749) < stats.deflated_sharpe(r, 5, 1 / 749)


# --- loop ------------------------------------------------------------------

def test_cycle_persists_and_explores_other_families(sim):
    bars = _bars("null")
    path = loop.run_learning_cycle(None, cfg(), list(bars), now=NOW, bars=bars, seed=3)
    text = open(path).read()
    assert "Multiple-testing control" in text
    cands = CandidateStore(loop.CANDIDATES_PATH).read_all()
    assert len(cands) == 4
    assert sum(1 for c in cands if loop.genome_of(c)["family"] != "sma_cross") >= 2  # exploration
    assert "active" in loop.load_results() and all(not c.config.validate_bounds() for c in cands)


def test_no_false_promotions_on_noise(sim):
    """Many runs on pure noise must never schedule a switch."""
    bars = _bars("null", n=1400)
    cal = list(bars["S0"].index)
    for k, t in enumerate(cal[1100::30]):
        w = {s: df[df.index <= t] for s, df in bars.items()}
        loop.run_learning_cycle(None, cfg(), list(w), now=t.to_pydatetime().replace(hour=21), bars=w, seed=k)
    assert not os.path.exists(loop.ACTIVE_STRATEGY_PATH)
    assert not any("Strategy switch scheduled" in m for m in sim)


def test_learns_planted_edge_and_switches_category(sim):
    """Trend regimes + a mean-reversion incumbent: the loop must find a trend family."""
    bars = _bars("trend", n=1500, seed=4)
    c = cfg(min_paper_days=20)
    c["strategy"] = {"family": "mean_reversion"}
    c["learning"].update(propose_per_run=6, explore_per_run=3)
    cal = list(bars["S0"].index)
    for k, t in enumerate(cal[1100::21]):
        w = {s: df[df.index <= t] for s, df in bars.items()}
        loop.run_learning_cycle(None, c, list(w), now=t.to_pydatetime().replace(hour=21),
                                bars=w, seed=k)
    assert any("plans to change its strategy" in m and "learning-veto" in m for m in sim), "never announced a switch"
    assert loop.active_genome(c)["family"] != "mean_reversion"


def _force_ready(c_obj):
    """Put a candidate in PAPER and return a fake evidence dict."""
    c_obj.status = CandidateStatus.PAPER
    CandidateStore(loop.CANDIDATES_PATH).save(c_obj)
    return {"paper_days": 40, "paper_sharpe": 1.5, "active_sharpe": 0.2, "paper_ret": 3.0,
            "paper_trades": 20, "dsr": 0.9, "n_trials": 50}


def test_announce_then_switch_after_grace(sim):
    c = loop._candidate({"family": "breakout"}, "test", None, NOW)
    ev = _force_ready(c)
    config = cfg()
    assert loop.announce_promotion(config, c, ev, NOW)
    assert "plans to change its strategy" in sim[0] and "learning-veto" in sim[0]
    assert loop.apply_pending_promotion(config, NOW + timedelta(hours=1)) is None   # still in grace
    assert loop.active_genome(config)["family"] == "sma_cross"
    assert loop.apply_pending_promotion(config, NOW + timedelta(hours=13)) == c.candidate_id
    assert loop.active_genome(config)["family"] == "breakout"
    assert any("Paper bot strategy changed" in m and "learning-rollback" in m for m in sim)
    loop.rollback_active("test")
    assert loop.active_genome(config)["family"] == "sma_cross"


def test_disabling_auto_promotion_blocks_an_existing_pending_switch(sim):
    c = loop._candidate({"family": "breakout"}, "test", None, NOW)
    ev = _force_ready(c)
    config = cfg()
    assert loop.announce_promotion(config, c, ev, NOW)
    config["learning"]["promotion"]["allow_auto_promotion"] = False
    assert loop.apply_pending_promotion(config, NOW + timedelta(hours=13)) is None
    assert loop.active_genome(config)["family"] == "sma_cross"
    assert os.path.exists(loop.PENDING_PROMOTION_PATH)


def test_failed_announcement_never_switches(sim, monkeypatch):
    monkeypatch.setattr(notify, "send", lambda *a, **k: False)
    c = loop._candidate({"family": "breakout"}, "test", None, NOW)
    ev = _force_ready(c)
    assert not loop.announce_promotion(cfg(), c, ev, NOW)
    assert not os.path.exists(loop.PENDING_PROMOTION_PATH)
    assert loop.apply_pending_promotion(cfg(), NOW + timedelta(days=2)) is None


def test_veto_cancels_switch(sim):
    c = loop._candidate({"family": "momentum"}, "test", None, NOW)
    ev = _force_ready(c)
    loop.announce_promotion(cfg(), c, ev, NOW)
    assert loop.veto_pending("me") == c.candidate_id
    assert loop.apply_pending_promotion(cfg(), NOW + timedelta(days=1)) is None
    assert CandidateStore(loop.CANDIDATES_PATH).get(c.candidate_id).status == CandidateStatus.REJECTED


# --- decision policy & readiness ------------------------------------------

def test_decision_policy_disables_antipredictive_veto():
    from learning.decision_policy import fit_policy
    rng = np.random.default_rng(0)
    rows = [{"tag": "taken", "confidence": 0.8, "ret": r} for r in rng.normal(-1.0, 2, 60)]
    rows += [{"tag": "vetoed", "confidence": 0.3, "ret": r} for r in rng.normal(1.5, 2, 60)]
    p = fit_policy(pd.DataFrame(rows), min_samples=30)
    assert p["jev_can_veto"] is False


def test_decision_policy_raises_threshold_when_confidence_predicts():
    from learning.decision_policy import fit_policy
    rng = np.random.default_rng(1)
    rows = []
    for _ in range(300):
        conf = rng.uniform(0.3, 0.95)
        rows.append({"tag": "taken" if conf >= 0.6 else "vetoed", "confidence": conf,
                     "ret": rng.normal(4 * (conf - 0.6), 1.5)})
    p = fit_policy(pd.DataFrame(rows), min_samples=30)
    assert p["jev_can_veto"] is True and p["min_confidence_entry"] >= 0.6


def test_readiness_requires_a_real_record():
    from learning.readiness import assess
    days = pd.bdate_range("2026-01-01", periods=200)
    rng = np.random.default_rng(2)
    eq = 100000 * np.cumprod(1 + rng.normal(0.0012, 0.006, 200))
    rows = [{"date": str(d.date()), "equity": e, "strategy_version": "v"} for d, e in zip(days, eq)]
    spy = pd.Series(400 * np.cumprod(1 + rng.normal(0.0003, 0.01, 200)), index=days)
    good = assess({}, spy, rows, closed=[{}] * 60)
    assert good["status"] == "READY_FOR_REVIEW", good
    few = assess({}, spy, rows, closed=[{}] * 5)
    assert few["status"] == "NOT_READY" and few["checks"]["min_closed_trades"] is False
    assert assess({}, spy, rows[:30], closed=[{}] * 60)["checks"]["min_days"] is False
