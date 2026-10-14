"""
Deterministic forecast evaluation — runs only when the forecast horizon has elapsed.

For each matured forecast, computes:
  - Actual return (price change from decision time to horizon end)
  - Benchmark return (buy-and-hold over the same period)
  - Forecast error (actual - expected)
  - Confidence calibration (binned by confidence level)
  - Costs (estimated transaction costs)

Uses only information available at the original decision time.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .forecasts import Forecast, ForecastStatus, ForecastStore

logger = logging.getLogger(__name__)


@dataclass
class ForecastEvaluation:
    """Result of evaluating a single forecast."""
    forecast_id: str
    symbol: str
    decision_time: str
    horizon_days: int
    expected_return_pct: float
    actual_return_pct: float
    benchmark_return_pct: float
    forecast_error_pct: float
    confidence: float
    calibration_error: float
    costs_pct: float
    net_return_pct: float  # actual - costs
    evaluated_at: str = ""

    def __post_init__(self):
        if not self.evaluated_at:
            self.evaluated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "forecast_id": self.forecast_id,
            "symbol": self.symbol,
            "decision_time": self.decision_time,
            "horizon_days": self.horizon_days,
            "expected_return_pct": round(self.expected_return_pct, 4),
            "actual_return_pct": round(self.actual_return_pct, 4),
            "benchmark_return_pct": round(self.benchmark_return_pct, 4),
            "forecast_error_pct": round(self.forecast_error_pct, 4),
            "confidence": round(self.confidence, 4),
            "calibration_error": round(self.calibration_error, 4),
            "costs_pct": round(self.costs_pct, 4),
            "net_return_pct": round(self.net_return_pct, 4),
            "evaluated_at": self.evaluated_at,
        }


@dataclass
class CalibrationBin:
    """Calibration statistics for a confidence bin."""
    bin_start: float  # 0.0, 0.1, 0.2, ...
    bin_end: float    # 0.1, 0.2, 0.3, ...
    count: int = 0
    avg_confidence: float = 0.0
    avg_actual: float = 0.0
    calibration_error: float = 0.0  # |avg_confidence - avg_actual|

    def to_dict(self) -> dict:
        return {
            "bin": f"{self.bin_start:.1f}-{self.bin_end:.1f}",
            "count": self.count,
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_actual": round(self.avg_actual, 4),
            "calibration_error": round(self.calibration_error, 4),
        }


@dataclass
class EvaluationBatch:
    """Batch of forecast evaluations with aggregate statistics."""
    evaluations: list[ForecastEvaluation] = field(default_factory=list)
    calibration_bins: list[CalibrationBin] = field(default_factory=list)

    @property
    def total_evaluated(self) -> int:
        return len(self.evaluations)

    @property
    def avg_forecast_error(self) -> float:
        if not self.evaluations:
            return 0.0
        return float(np.mean([e.forecast_error_pct for e in self.evaluations]))

    @property
    def avg_calibration_error(self) -> float:
        if not self.calibration_bins:
            return 0.0
        # Weighted average by bin count
        total_count = sum(b.count for b in self.calibration_bins)
        if total_count == 0:
            return 0.0
        return sum(b.calibration_error * b.count for b in self.calibration_bins) / total_count

    @property
    def avg_net_return(self) -> float:
        if not self.evaluations:
            return 0.0
        return float(np.mean([e.net_return_pct for e in self.evaluations]))

    @property
    def hit_rate(self) -> float:
        """Percentage of forecasts where actual direction matched expected."""
        if not self.evaluations:
            return 0.0
        hits = sum(
            1 for e in self.evaluations
            if (e.expected_return_pct > 0 and e.actual_return_pct > 0)
            or (e.expected_return_pct < 0 and e.actual_return_pct < 0)
            or (e.expected_return_pct == 0 and abs(e.actual_return_pct) < 0.1)
        )
        return hits / len(self.evaluations)

    def to_dict(self) -> dict:
        return {
            "total_evaluated": self.total_evaluated,
            "avg_forecast_error": round(self.avg_forecast_error, 4),
            "avg_calibration_error": round(self.avg_calibration_error, 4),
            "avg_net_return": round(self.avg_net_return, 4),
            "hit_rate": round(self.hit_rate, 4),
            "calibration_bins": [b.to_dict() for b in self.calibration_bins],
            "evaluations": [e.to_dict() for e in self.evaluations],
        }


class ForecastEvaluator:
    """
    Evaluates forecasts deterministically once their horizon has elapsed.

    Uses only information available at the original decision time.
    No look-ahead: the evaluator fetches the actual price at the horizon
    end date and computes the realized return.
    """

    # Cost model defaults
    DEFAULT_COMMISSION_PER_SHARE = 0.005
    DEFAULT_MIN_COMMISSION = 1.0
    DEFAULT_SLIPPAGE_PCT = 0.001
    DEFAULT_SPREAD_PCT = 0.0005

    def __init__(
        self,
        commission_per_share: float = DEFAULT_COMMISSION_PER_SHARE,
        min_commission: float = DEFAULT_MIN_COMMISSION,
        slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
        spread_pct: float = DEFAULT_SPREAD_PCT,
    ):
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.slippage_pct = slippage_pct
        self.spread_pct = spread_pct

    def evaluate(
        self,
        forecast: Forecast,
        df: pd.DataFrame,
        now: Optional[datetime] = None,
    ) -> Optional[ForecastEvaluation]:
        """
        Evaluate a single forecast against actual price data.

        Parameters
        ----------
        forecast : Forecast
            The forecast to evaluate.
        df : pd.DataFrame
            Price data with DatetimeIndex and 'Close' column.
            Must cover the full horizon period.
        now : datetime, optional
            Current time for maturity check.

        Returns
        -------
        ForecastEvaluation or None if the forecast hasn't matured.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        if not forecast.is_matured(now):
            return None

        if df.empty or "Close" not in df.columns:
            return None

        # Find the decision-time price
        decision_dt = datetime.fromisoformat(forecast.decision_time)
        if decision_dt.tzinfo is None:
            decision_dt = decision_dt.replace(tzinfo=timezone.utc)

        # Find closest price at or after decision time
        # Handle both date-only and datetime indices
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize(timezone.utc)
        decision_mask = df.index >= decision_dt
        if not decision_mask.any():
            return None
        decision_price = float(df.loc[decision_mask, "Close"].iloc[0])

        # Find the horizon-end price
        horizon_end = forecast.horizon_end
        if horizon_end is None:
            return None

        # Find closest price at or after horizon end
        horizon_mask = df.index >= horizon_end
        if horizon_mask.any():
            horizon_price = float(df.loc[horizon_mask, "Close"].iloc[0])
        else:
            # Use last available price if horizon end is beyond data
            horizon_price = float(df["Close"].iloc[-1])

        # Compute returns
        actual_return = (horizon_price - decision_price) / decision_price * 100

        # Benchmark: buy-and-hold over the same period
        benchmark_return = actual_return  # Same period, same instrument

        # Forecast error
        forecast_error = actual_return - forecast.expected_return_pct

        # Calibration error: |confidence - actual_direction_score|
        # For a positive expected return, actual_direction_score = 1 if actual > 0, else 0
        if forecast.expected_return_pct > 0:
            actual_direction = 1.0 if actual_return > 0 else 0.0
        elif forecast.expected_return_pct < 0:
            actual_direction = 1.0 if actual_return < 0 else 0.0
        else:
            actual_direction = 1.0 if abs(actual_return) < 0.1 else 0.0
        calibration_error = abs(forecast.confidence - actual_direction)

        # Cost estimation
        costs_pct = self._estimate_costs(decision_price)

        # Net return
        net_return = actual_return - costs_pct

        return ForecastEvaluation(
            forecast_id=forecast.forecast_id,
            symbol=forecast.symbol,
            decision_time=forecast.decision_time,
            horizon_days=forecast.horizon_days,
            expected_return_pct=forecast.expected_return_pct,
            actual_return_pct=round(actual_return, 4),
            benchmark_return_pct=round(benchmark_return, 4),
            forecast_error_pct=round(forecast_error, 4),
            confidence=forecast.confidence,
            calibration_error=round(calibration_error, 4),
            costs_pct=round(costs_pct, 4),
            net_return_pct=round(net_return, 4),
        )

    def evaluate_batch(
        self,
        forecasts: list[Forecast],
        price_data: dict[str, pd.DataFrame],
        now: Optional[datetime] = None,
    ) -> EvaluationBatch:
        """
        Evaluate a batch of forecasts.

        Parameters
        ----------
        forecasts : list[Forecast]
            Forecasts to evaluate.
        price_data : dict[str, pd.DataFrame]
            Price data keyed by symbol.
        now : datetime, optional
            Current time for maturity check.

        Returns
        -------
        EvaluationBatch with all evaluations and calibration statistics.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        evaluations = []
        for forecast in forecasts:
            if not forecast.is_matured(now):
                continue

            df = price_data.get(forecast.symbol)
            if df is None or df.empty:
                continue

            result = self.evaluate(forecast, df, now)
            if result is not None:
                evaluations.append(result)

        # Compute calibration bins
        calibration_bins = self._compute_calibration_bins(evaluations)

        return EvaluationBatch(
            evaluations=evaluations,
            calibration_bins=calibration_bins,
        )

    def _estimate_costs(self, price: float) -> float:
        """Estimate transaction costs as a percentage of notional."""
        # Assume ~$5000 position
        position = 5000.0
        qty = position / price if price > 0 else 1

        commission = max(self.min_commission, self.commission_per_share * qty)
        slippage = position * self.slippage_pct * 2  # entry + exit
        spread = position * self.spread_pct * 2

        total_costs = commission + slippage + spread
        return (total_costs / position) * 100

    def _compute_calibration_bins(
        self, evaluations: list[ForecastEvaluation]
    ) -> list[CalibrationBin]:
        """Compute calibration statistics for 10 confidence bins."""
        bins = []
        for i in range(10):
            bin_start = round(i * 0.1, 1)
            bin_end = round((i + 1) * 0.1, 1)
            bin_evals = [
                e for e in evaluations
                if bin_start <= e.confidence < bin_end
            ]
            # Include the upper bound in the last bin
            if i == 9:
                bin_evals = [
                    e for e in evaluations
                    if bin_start <= e.confidence <= bin_end
                ]

            count = len(bin_evals)
            if count > 0:
                avg_conf = float(np.mean([e.confidence for e in bin_evals]))
                # For calibration, we use the actual direction score
                avg_actual = float(np.mean([
                    1.0 if (e.expected_return_pct > 0 and e.actual_return_pct > 0)
                    or (e.expected_return_pct < 0 and e.actual_return_pct < 0)
                    or (e.expected_return_pct == 0 and abs(e.actual_return_pct) < 0.1)
                    else 0.0
                    for e in bin_evals
                ]))
                cal_error = abs(avg_conf - avg_actual)
            else:
                avg_conf = 0.0
                avg_actual = 0.0
                cal_error = 0.0

            bins.append(CalibrationBin(
                bin_start=bin_start,
                bin_end=bin_end,
                count=count,
                avg_confidence=round(avg_conf, 4),
                avg_actual=round(avg_actual, 4),
                calibration_error=round(cal_error, 4),
            ))

        return bins
