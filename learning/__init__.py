"""
Recursive Learning and Strategy Selection System.

This package provides a controlled, auditable process for evaluating
alternative models and tactics over time. Candidates are promoted only
when there is credible evidence they improve performance.

Key principles:
  - Decision-time trading is separate from learning
  - Learning runs on recorded historical data and paper-trading results
  - Candidate strategies never modify the active strategy directly
  - No automatic promotion — operator approval required
  - The learning service has NO order credentials or order-placement tools
"""

from .candidates import Candidate, CandidateStore, CandidateConfig
from .evaluation import WalkForwardEvaluator, EvaluationResult, BaselineResult
from .promotion import PromotionGate, PromotionDecision
from .model_selection import ModelSelector, SelectionResult
from .audit import LearningAuditLog
from .forecasts import Forecast, ForecastStore, ForecastStatus, generate_forecast_id
from .forecast_evaluator import ForecastEvaluator, ForecastEvaluation, EvaluationBatch, CalibrationBin
from .challenger_comparison import ChallengerComparison, ComparisonResult, ComparisonConfig

__all__ = [
    "Candidate",
    "CandidateStore",
    "CandidateConfig",
    "WalkForwardEvaluator",
    "EvaluationResult",
    "BaselineResult",
    "PromotionGate",
    "PromotionDecision",
    "ModelSelector",
    "SelectionResult",
    "LearningAuditLog",
    "Forecast",
    "ForecastStore",
    "ForecastStatus",
    "generate_forecast_id",
    "ForecastEvaluator",
    "ForecastEvaluation",
    "EvaluationBatch",
    "CalibrationBin",
    "ChallengerComparison",
    "ComparisonResult",
    "ComparisonConfig",
]
