"""Tests for challenger comparison."""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BOT_DIR)

from learning.candidates import Candidate, CandidateConfig, CandidateStatus
from learning.challenger_comparison import (
    ChallengerComparison, ComparisonResult, ComparisonConfig,
)


@pytest.fixture
def sample_candidate():
    return Candidate(
        candidate_id="test-challenger-001",
        version="v1.0",
        hypothesis="Test challenger",
        intended_conditions="Trending markets",
        required_data="Daily OHLCV",
        abstain_rule="Abstain when SMAs converge",
        config=CandidateConfig(sma_short=10, sma_long=30),
    )


@pytest.fixture
def sample_df():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=400, freq="D")
    close = 100 + np.cumsum(np.random.randn(400) * 0.5)
    df = pd.DataFrame({
        "Open": close + np.random.randn(400) * 0.1,
        "High": close + abs(np.random.randn(400) * 0.5),
        "Low": close - abs(np.random.randn(400) * 0.5),
        "Close": close,
        "Volume": np.random.randint(1000000, 5000000, 400),
    }, index=dates)
    return df


@pytest.fixture
def comparison():
    return ChallengerComparison(ComparisonConfig())


class TestComparisonResult:
    def test_create_result(self):
        result = ComparisonResult(
            challenger_id="test-001",
            challenger_version="v1.0",
            active_version="v1.1",
            symbol="AAPL",
            period="validation",
        )
        assert result.challenger_id == "test-001"
        assert result.challenger_passes_gates is False
        assert result.rejection_reasons == []

    def test_to_dict(self):
        result = ComparisonResult(
            challenger_id="test-001",
            challenger_version="v1.0",
            active_version="v1.1",
            symbol="AAPL",
            period="validation",
        )
        d = result.to_dict()
        assert d["challenger_id"] == "test-001"
        assert "challenger_passes_gates" in d


class TestChallengerComparison:
    def test_compare_returns_result(self, comparison, sample_candidate, sample_df):
        from strategies.v1_1 import SMACrossV1_1

        challenger_strategy = SMACrossV1_1(sma_short=10, sma_long=30)
        active_strategy = SMACrossV1_1(sma_short=20, sma_long=50)

        result = comparison.compare(
            challenger=sample_candidate,
            challenger_strategy=challenger_strategy,
            active_strategy=active_strategy,
            df=sample_df,
            symbol="AAPL",
            period="validation",
        )
        assert isinstance(result, ComparisonResult)
        assert result.challenger_id == sample_candidate.candidate_id
        assert result.symbol == "AAPL"

    def test_compare_empty_df(self, comparison, sample_candidate):
        from strategies.v1_1 import SMACrossV1_1

        challenger_strategy = SMACrossV1_1(sma_short=10, sma_long=30)
        active_strategy = SMACrossV1_1(sma_short=20, sma_long=50)

        result = comparison.compare(
            challenger=sample_candidate,
            challenger_strategy=challenger_strategy,
            active_strategy=active_strategy,
            df=pd.DataFrame(),
            symbol="AAPL",
            period="validation",
        )
        assert result.rejection_reasons == ["empty_dataframe"]

    def test_compare_all_periods(self, comparison, sample_candidate, sample_df):
        from strategies.v1_1 import SMACrossV1_1

        challenger_strategy = SMACrossV1_1(sma_short=10, sma_long=30)
        active_strategy = SMACrossV1_1(sma_short=20, sma_long=50)

        results = comparison.compare_all_periods(
            challenger=sample_candidate,
            challenger_strategy=challenger_strategy,
            active_strategy=active_strategy,
            df=sample_df,
            symbol="AAPL",
        )
        assert len(results) > 0
        for r in results:
            assert isinstance(r, ComparisonResult)
            assert r.symbol == "AAPL"

    def test_config_defaults(self):
        config = ComparisonConfig()
        assert config.train_days == 252
        assert config.validation_days == 126
        assert config.out_of_sample_days == 63
        assert config.min_trade_frequency == 0.011
