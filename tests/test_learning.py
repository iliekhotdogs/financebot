"""Tests for the recursive learning system."""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Ensure bot root is on path
BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BOT_DIR)

from learning.candidates import (
    Candidate, CandidateStore, CandidateConfig, CandidateStatus,
)
from learning.evaluation import WalkForwardEvaluator, EvaluationResult
from learning.promotion import PromotionGate, PromotionGateConfig, PromotionDecision
from learning.model_selection import ModelSelector, SelectionConfig, SelectionMetric, SelectionResult
from learning.audit import LearningAuditLog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config():
    return CandidateConfig(
        sma_short=10,
        sma_long=30,
        atr_period=14,
        atr_mult_initial=3.0,
    )


@pytest.fixture
def sample_candidate(sample_config):
    return Candidate(
        candidate_id="test-cand-001",
        version="1.0",
        hypothesis="Test hypothesis",
        intended_conditions="Trending markets",
        required_data="Daily OHLCV",
        abstain_rule="Abstain when SMAs converge",
        config=sample_config,
    )


@pytest.fixture
def sample_df():
    """Create a sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    close = 100 + np.cumsum(np.random.randn(300) * 0.5)
    df = pd.DataFrame({
        "Open": close + np.random.randn(300) * 0.1,
        "High": close + abs(np.random.randn(300) * 0.5),
        "Low": close - abs(np.random.randn(300) * 0.5),
        "Close": close,
        "Volume": np.random.randint(1000000, 5000000, 300),
    }, index=dates)
    return df


@pytest.fixture
def store(tmp_path):
    path = str(tmp_path / "candidates.csv")
    return CandidateStore(path=path)


@pytest.fixture
def audit_log(tmp_path):
    path = str(tmp_path / "learning_audit.csv")
    return LearningAuditLog(path=path)


# ---------------------------------------------------------------------------
# Candidate tests
# ---------------------------------------------------------------------------

class TestCandidateConfig:
    def test_default_bounds(self):
        config = CandidateConfig()
        assert config.sma_short == 20
        assert config.sma_long == 50
        assert config.validate_bounds() == []

    def test_custom_params_within_bounds(self):
        config = CandidateConfig(sma_short=10, sma_long=30)
        assert config.validate_bounds() == []

    def test_sma_short_greater_than_long_fails(self):
        config = CandidateConfig(sma_short=50, sma_long=30)
        errors = config.validate_bounds()
        assert len(errors) > 0
        assert any("sma_short" in e for e in errors)

    def test_sma_outside_bounds_fails(self):
        config = CandidateConfig(sma_short=3, sma_long=300)
        errors = config.validate_bounds()
        assert len(errors) >= 2

    def test_atr_mult_outside_bounds_fails(self):
        config = CandidateConfig(atr_mult_initial=10.0)
        errors = config.validate_bounds()
        assert any("atr_mult" in e for e in errors)


class TestCandidate:
    def test_create_candidate(self, sample_candidate):
        assert sample_candidate.candidate_id == "test-cand-001"
        assert sample_candidate.status == CandidateStatus.PROPOSED
        assert sample_candidate.code_hash != ""
        assert sample_candidate.created_at != ""

    def test_code_hash_deterministic(self, sample_config):
        c1 = Candidate(
            candidate_id="test-1", version="1.0",
            hypothesis="h", intended_conditions="ic",
            required_data="rd", abstain_rule="ar",
            config=sample_config,
        )
        c2 = Candidate(
            candidate_id="test-2", version="1.0",
            hypothesis="h", intended_conditions="ic",
            required_data="rd", abstain_rule="ar",
            config=sample_config,
        )
        assert c1.code_hash == c2.code_hash

    def test_code_hash_changes_with_config(self, sample_config):
        c1 = Candidate(
            candidate_id="test-1", version="1.0",
            hypothesis="h", intended_conditions="ic",
            required_data="rd", abstain_rule="ar",
            config=sample_config,
        )
        config2 = CandidateConfig(sma_short=15, sma_long=40)
        c2 = Candidate(
            candidate_id="test-2", version="1.0",
            hypothesis="h", intended_conditions="ic",
            required_data="rd", abstain_rule="ar",
            config=config2,
        )
        assert c1.code_hash != c2.code_hash

    def test_to_dict(self, sample_candidate):
        d = sample_candidate.to_dict()
        assert d["candidate_id"] == "test-cand-001"
        assert d["status"] == "proposed"
        assert "config" in d

    def test_from_dict(self, sample_candidate):
        d = sample_candidate.to_dict()
        restored = Candidate.from_dict(d)
        assert restored.candidate_id == sample_candidate.candidate_id
        assert restored.status == sample_candidate.status
        assert restored.config.sma_short == sample_candidate.config.sma_short


class TestCandidateStore:
    def test_save_and_get(self, store, sample_candidate):
        store.save(sample_candidate)
        retrieved = store.get("test-cand-001")
        assert retrieved is not None
        assert retrieved.candidate_id == "test-cand-001"
        assert retrieved.status == CandidateStatus.PROPOSED

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None

    def test_read_all(self, store, sample_candidate):
        store.save(sample_candidate)
        all_candidates = store.read_all()
        assert len(all_candidates) == 1

    def test_update_existing(self, store, sample_candidate):
        store.save(sample_candidate)
        sample_candidate.status = CandidateStatus.EVALUATING
        store.save(sample_candidate)
        all_candidates = store.read_all()
        assert len(all_candidates) == 1
        assert all_candidates[0].status == CandidateStatus.EVALUATING

    def test_get_by_status(self, store, sample_candidate):
        store.save(sample_candidate)
        proposed = store.get_by_status(CandidateStatus.PROPOSED)
        assert len(proposed) == 1
        evaluating = store.get_by_status(CandidateStatus.EVALUATING)
        assert len(evaluating) == 0


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

class TestWalkForwardEvaluator:
    def test_evaluate_returns_result(self, sample_df):
        from strategies.v1_1 import SMACrossV1_1
        evaluator = WalkForwardEvaluator()
        strategy = SMACrossV1_1(sma_short=10, sma_long=30)
        result = evaluator.evaluate("test-cand", strategy, sample_df, "AAPL", "validation")
        assert isinstance(result, EvaluationResult)
        assert result.candidate_id == "test-cand"
        assert result.symbol == "AAPL"
        assert result.period == "validation"

    def test_data_integrity_empty_df(self):
        evaluator = WalkForwardEvaluator()
        from strategies.v1_1 import SMACrossV1_1
        strategy = SMACrossV1_1(sma_short=10, sma_long=30)
        empty_df = pd.DataFrame()
        result = evaluator.evaluate("test", strategy, empty_df, "AAPL", "validation")
        assert not result.data_quality_passed
        assert len(result.data_quality_issues) > 0

    def test_data_integrity_insufficient_data(self):
        evaluator = WalkForwardEvaluator()
        from strategies.v1_1 import SMACrossV1_1
        strategy = SMACrossV1_1(sma_short=10, sma_long=30)
        small_df = pd.DataFrame({
            "Open": [100] * 10,
            "High": [101] * 10,
            "Low": [99] * 10,
            "Close": [100] * 10,
            "Volume": [1000] * 10,
        })
        result = evaluator.evaluate("test", strategy, small_df, "AAPL", "validation")
        assert not result.data_quality_passed

    def test_baselines_computed(self, sample_df):
        evaluator = WalkForwardEvaluator()
        from strategies.v1_1 import SMACrossV1_1
        strategy = SMACrossV1_1(sma_short=10, sma_long=30)
        result = evaluator.evaluate("test", strategy, sample_df, "AAPL", "validation")
        # Baselines should be computed (even if zero)
        assert hasattr(result, "baseline_sma_return_pct")
        assert hasattr(result, "baseline_buyhold_return_pct")

    def test_cost_modeling_applied(self, sample_df):
        evaluator = WalkForwardEvaluator()
        from strategies.v1_1 import SMACrossV1_1
        strategy = SMACrossV1_1(sma_short=10, sma_long=30)
        result = evaluator.evaluate("test", strategy, sample_df, "AAPL", "validation")
        # Cost fields should exist
        assert hasattr(result, "total_fees")
        assert hasattr(result, "total_slippage")
        assert hasattr(result, "total_spread_cost")

    def test_confidence_interval(self):
        evaluator = WalkForwardEvaluator()
        returns = [1.0, 2.0, -1.0, 0.5, 1.5, -0.5, 2.0, 1.0, -1.0, 0.5]
        ci = evaluator._compute_confidence_interval(returns)
        assert len(ci) == 2
        assert ci[0] < ci[1]

    def test_confidence_interval_insufficient(self):
        evaluator = WalkForwardEvaluator()
        ci = evaluator._compute_confidence_interval([1.0])
        assert ci == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Promotion tests
# ---------------------------------------------------------------------------

class TestPromotionGate:
    def test_assess_for_paper_no_results(self, sample_candidate):
        gate = PromotionGate(PromotionGateConfig())
        assessment = gate.assess_for_paper(sample_candidate, [])
        assert assessment.decision == PromotionDecision.NEEDS_MORE_DATA

    def test_assess_for_paper_failed_gates(self, sample_candidate):
        gate = PromotionGate(PromotionGateConfig())
        # Create evaluation results that fail gates
        from learning.evaluation import EvaluationResult
        result = EvaluationResult(
            candidate_id="test-cand-001",
            symbol="AAPL",
            period="validation",
            start_date="2023-01-01",
            end_date="2023-06-30",
            total_trades=2,  # too few
            sharpe_ratio=-1.0,  # poor
            passed_gates=False,
        )
        assessment = gate.assess_for_paper(sample_candidate, [result])
        assert assessment.decision == PromotionDecision.REJECTED

    def test_assess_for_promotion_insufficient_paper(self, sample_candidate):
        gate = PromotionGate(PromotionGateConfig())
        paper_results = {
            "days_in_paper": 5,  # too few
            "total_trades": 3,
            "symbols_traded": 1,
            "sharpe_ratio": 0.8,
            "max_drawdown_pct": 5.0,
            "net_return_pct": 2.0,
        }
        active_results = {
            "sharpe_ratio": 0.5,
            "net_return_pct": 1.0,
        }
        assessment = gate.assess_for_promotion(sample_candidate, paper_results, active_results)
        assert assessment.decision == PromotionDecision.NEEDS_MORE_DATA
        assert len(assessment.gates_failed) > 0

    def test_assess_for_promotion_ready(self, sample_candidate):
        gate = PromotionGate(PromotionGateConfig())
        paper_results = {
            "days_in_paper": 45,
            "total_trades": 15,
            "symbols_traded": 3,
            "sharpe_ratio": 1.2,
            "max_drawdown_pct": 5.0,
            "net_return_pct": 3.0,
        }
        active_results = {
            "sharpe_ratio": 0.5,
            "net_return_pct": 1.0,
        }
        assessment = gate.assess_for_promotion(sample_candidate, paper_results, active_results)
        assert assessment.decision == PromotionDecision.READY_FOR_PROMOTION
        assert len(assessment.gates_failed) == 0

    def test_auto_promotion_never_enabled(self):
        config = PromotionGateConfig()
        assert config.allow_auto_promotion is False


# ---------------------------------------------------------------------------
# Model selection tests
# ---------------------------------------------------------------------------

class TestModelSelector:
    def test_select_no_candidates(self):
        selector = ModelSelector(SelectionConfig())
        result = selector.select([], {})
        assert result.selected_candidate_id is None
        assert result.active_version_retained is True

    def test_select_retains_active_when_no_gates_passed(self, sample_candidate):
        selector = ModelSelector(SelectionConfig())
        # No evaluation results
        result = selector.select([sample_candidate], {})
        assert result.selected_candidate_id is None
        assert result.active_version_retained is True

    def test_select_passing_candidate(self, sample_candidate):
        selector = ModelSelector(SelectionConfig())
        from learning.evaluation import EvaluationResult
        eval_results = {
            "test-cand-001": [
                EvaluationResult(
                    candidate_id="test-cand-001", symbol="AAPL", period="validation",
                    start_date="2023-01-01", end_date="2023-06-30",
                    net_return_pct=5.0, sharpe_ratio=1.5, max_drawdown_pct=5.0,
                    total_trades=15, passed_gates=True,
                    data_quality_passed=True, look_ahead_check_passed=True,
                ),
                EvaluationResult(
                    candidate_id="test-cand-001", symbol="AAPL", period="out_of_sample",
                    start_date="2023-07-01", end_date="2023-09-30",
                    net_return_pct=3.0, sharpe_ratio=1.2, max_drawdown_pct=3.0,
                    total_trades=12, passed_gates=True,
                    data_quality_passed=True, look_ahead_check_passed=True,
                ),
            ]
        }
        result = selector.select([sample_candidate], eval_results)
        assert result.selected_candidate_id == "test-cand-001"
        assert result.active_version_retained is False

    def test_select_rejects_high_drawdown(self, sample_candidate):
        selector = ModelSelector(SelectionConfig(max_drawdown_pct=5.0))
        from learning.evaluation import EvaluationResult
        eval_results = {
            "test-cand-001": [
                EvaluationResult(
                    candidate_id="test-cand-001", symbol="AAPL", period="validation",
                    start_date="2023-01-01", end_date="2023-06-30",
                    net_return_pct=20.0, sharpe_ratio=3.0, max_drawdown_pct=15.0,  # too high
                    total_trades=15, passed_gates=True,
                    data_quality_passed=True, look_ahead_check_passed=True,
                ),
                EvaluationResult(
                    candidate_id="test-cand-001", symbol="AAPL", period="out_of_sample",
                    start_date="2023-07-01", end_date="2023-09-30",
                    net_return_pct=15.0, sharpe_ratio=2.5, max_drawdown_pct=12.0,
                    total_trades=12, passed_gates=True,
                    data_quality_passed=True, look_ahead_check_passed=True,
                ),
            ]
        }
        result = selector.select([sample_candidate], eval_results)
        assert result.selected_candidate_id is None
        assert result.active_version_retained is True


# ---------------------------------------------------------------------------
# Audit log tests
# ---------------------------------------------------------------------------

class TestLearningAuditLog:
    def test_log_event(self, audit_log):
        row = audit_log.log_event(
            event_type="candidate_created",
            candidate_id="test-001",
            candidate_version="1.0",
            details={"hypothesis": "test"},
            rationale="Test creation",
        )
        assert row[1] == "candidate_created"
        assert row[2] == "test-001"

    def test_read_all(self, audit_log):
        audit_log.log_event("candidate_created", "test-001", "1.0")
        audit_log.log_event("evaluation_complete", "test-001", "1.0")
        events = audit_log.read_all()
        assert len(events) == 2

    def test_get_events_for_candidate(self, audit_log):
        audit_log.log_event("candidate_created", "test-001", "1.0")
        audit_log.log_event("candidate_created", "test-002", "1.0")
        audit_log.log_event("evaluation_complete", "test-001", "1.0")
        events = audit_log.get_events_for_candidate("test-001")
        assert len(events) == 2

    def test_get_events_by_type(self, audit_log):
        audit_log.log_event("candidate_created", "test-001", "1.0")
        audit_log.log_event("evaluation_complete", "test-001", "1.0")
        events = audit_log.get_events_by_type("candidate_created")
        assert len(events) == 1
