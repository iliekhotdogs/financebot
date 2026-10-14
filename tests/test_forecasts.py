"""Tests for forecast recording and evaluation."""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BOT_DIR)

from learning.forecasts import (
    Forecast, ForecastStore, ForecastStatus, generate_forecast_id,
    FORECAST_HEADERS,
)
from learning.forecast_evaluator import (
    ForecastEvaluator, ForecastEvaluation, EvaluationBatch, CalibrationBin,
)


@pytest.fixture
def sample_forecast():
    return Forecast(
        forecast_id="fc-AAPL-20261001120000-abc123",
        decision_time="2026-10-01T12:00:00+00:00",
        symbol="AAPL",
        horizon_days=1,
        expected_return_pct=0.5,
        confidence=0.65,
        thesis="SMA cross bullish signal",
        source_publication_times='["2026-10-01T11:55:00+00:00"]',
        strategy_version="v1.1",
        data_snapshot_id="snap-001",
    )


@pytest.fixture
def sample_df():
    """Create sample price data covering the forecast horizon."""
    dates = pd.date_range("2026-09-30", periods=10, freq="D")
    close = 100 + np.cumsum(np.random.randn(10) * 0.5)
    df = pd.DataFrame({
        "Open": close + np.random.randn(10) * 0.1,
        "High": close + abs(np.random.randn(10) * 0.5),
        "Low": close - abs(np.random.randn(10) * 0.5),
        "Close": close,
        "Volume": np.random.randint(1000000, 5000000, 10),
    }, index=dates)
    return df


@pytest.fixture
def store(tmp_path):
    path = str(tmp_path / "forecasts.csv")
    return ForecastStore(path=path)


class TestForecast:
    def test_create_forecast(self, sample_forecast):
        assert sample_forecast.forecast_id == "fc-AAPL-20261001120000-abc123"
        assert sample_forecast.symbol == "AAPL"
        assert sample_forecast.status == ForecastStatus.PENDING
        assert sample_forecast.created_at != ""

    def test_horizon_end(self, sample_forecast):
        end = sample_forecast.horizon_end
        assert end is not None
        assert end > datetime.fromisoformat(sample_forecast.decision_time)

    def test_is_matured(self, sample_forecast):
        # Decision time is 2026-10-01T12:00, horizon 1 day
        # horizon_end = 2026-10-01 + 2 calendar days = 2026-10-03T12:00
        future = datetime(2026, 10, 4, tzinfo=timezone.utc)
        assert sample_forecast.is_matured(future)

        past = datetime(2026, 10, 1, 13, 0, tzinfo=timezone.utc)
        assert not sample_forecast.is_matured(past)

    def test_to_dict(self, sample_forecast):
        d = sample_forecast.to_dict()
        assert d["forecast_id"] == "fc-AAPL-20261001120000-abc123"
        assert d["status"] == "pending"

    def test_from_dict(self, sample_forecast):
        d = sample_forecast.to_dict()
        restored = Forecast.from_dict(d)
        assert restored.forecast_id == sample_forecast.forecast_id
        assert restored.status == sample_forecast.status


class TestForecastStore:
    def test_record_and_get(self, store, sample_forecast):
        store.record(sample_forecast)
        retrieved = store.get(sample_forecast.forecast_id)
        assert retrieved is not None
        assert retrieved.forecast_id == sample_forecast.forecast_id
        assert retrieved.symbol == "AAPL"

    def test_duplicate_forecast_raises(self, store, sample_forecast):
        store.record(sample_forecast)
        with pytest.raises(ValueError, match="already exists"):
            store.record(sample_forecast)

    def test_read_all(self, store, sample_forecast):
        store.record(sample_forecast)
        all_forecasts = store.read_all()
        assert len(all_forecasts) == 1

    def test_get_by_status(self, store, sample_forecast):
        store.record(sample_forecast)
        pending = store.get_by_status(ForecastStatus.PENDING)
        assert len(pending) == 1
        evaluated = store.get_by_status(ForecastStatus.EVALUATED)
        assert len(evaluated) == 0

    def test_get_by_symbol(self, store, sample_forecast):
        store.record(sample_forecast)
        aapl = store.get_by_symbol("AAPL")
        assert len(aapl) == 1
        msft = store.get_by_symbol("MSFT")
        assert len(msft) == 0

    def test_get_matured(self, store, sample_forecast):
        store.record(sample_forecast)
        future = datetime(2026, 10, 4, tzinfo=timezone.utc)
        matured = store.get_matured(future)
        assert len(matured) == 1

    def test_update_status(self, store, sample_forecast):
        store.record(sample_forecast)
        result = store.update_status(sample_forecast.forecast_id, ForecastStatus.EVALUATED)
        assert result is True
        retrieved = store.get(sample_forecast.forecast_id)
        assert retrieved.status == ForecastStatus.EVALUATED

    def test_update_status_not_found(self, store):
        result = store.update_status("nonexistent", ForecastStatus.EVALUATED)
        assert result is False

    def test_get_stats(self, store, sample_forecast):
        store.record(sample_forecast)
        stats = store.get_stats()
        assert stats["total"] == 1
        assert stats["by_symbol"]["AAPL"] == 1


class TestForecastEvaluator:
    def test_evaluate_returns_result(self, sample_forecast, sample_df):
        evaluator = ForecastEvaluator()
        future = datetime(2026, 10, 4, tzinfo=timezone.utc)
        result = evaluator.evaluate(sample_forecast, sample_df, now=future)
        assert result is not None
        assert isinstance(result, ForecastEvaluation)
        assert result.forecast_id == sample_forecast.forecast_id

    def test_evaluate_not_matured(self, sample_forecast, sample_df):
        evaluator = ForecastEvaluator()
        past = datetime(2026, 10, 1, 13, 0, tzinfo=timezone.utc)
        result = evaluator.evaluate(sample_forecast, sample_df, now=past)
        assert result is None

    def test_evaluate_empty_df(self, sample_forecast):
        evaluator = ForecastEvaluator()
        future = datetime(2026, 10, 3, tzinfo=timezone.utc)
        result = evaluator.evaluate(sample_forecast, pd.DataFrame(), now=future)
        assert result is None

    def test_evaluate_batch(self, sample_forecast, sample_df):
        evaluator = ForecastEvaluator()
        future = datetime(2026, 10, 4, tzinfo=timezone.utc)
        batch = evaluator.evaluate_batch(
            [sample_forecast],
            {"AAPL": sample_df},
            now=future,
        )
        assert isinstance(batch, EvaluationBatch)
        assert batch.total_evaluated == 1
        assert len(batch.calibration_bins) == 10

    def test_calibration_bins(self):
        evaluator = ForecastEvaluator()
        # Create mock evaluations
        evals = []
        for i in range(20):
            evals.append(ForecastEvaluation(
                forecast_id=f"fc-{i}",
                symbol="AAPL",
                decision_time="2026-10-01T12:00:00+00:00",
                horizon_days=1,
                expected_return_pct=0.5,
                actual_return_pct=0.3,
                benchmark_return_pct=0.3,
                forecast_error_pct=-0.2,
                confidence=0.6,
                calibration_error=0.4,
                costs_pct=0.1,
                net_return_pct=0.2,
            ))
        bins = evaluator._compute_calibration_bins(evals)
        assert len(bins) == 10
        # Bin 0.6-0.7 should have 20 items
        bin_06 = [b for b in bins if abs(b.bin_start - 0.6) < 0.01][0]
        assert bin_06.count == 20


class TestGenerateForecastId:
    def test_unique_ids(self):
        id1 = generate_forecast_id("AAPL", "2026-10-01T12:00:00+00:00", "thesis1")
        id2 = generate_forecast_id("AAPL", "2026-10-01T12:00:00+00:00", "thesis2")
        assert id1 != id2

    def test_contains_symbol(self):
        fid = generate_forecast_id("AAPL", "2026-10-01T12:00:00+00:00", "thesis")
        assert "AAPL" in fid
