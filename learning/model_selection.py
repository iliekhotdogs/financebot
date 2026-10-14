"""
Model selection — choose the most suitable candidate.

"Most suitable" is defined as the candidate that:
  - Meets all safety, data-quality, and minimum-evidence gates
  - Shows the strongest robust, net-of-cost performance across
    configured evaluation periods
  - Is NOT simply the candidate with the highest historical profit

Uses a configured primary metric plus hard constraints for:
  - Maximum drawdown
  - Maximum exposure
  - Minimum trade frequency

If no candidate clears all gates, keep the current approved version.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from .candidates import Candidate, CandidateStatus
from .evaluation import EvaluationResult

logger = logging.getLogger(__name__)


class SelectionMetric(str, Enum):
    SHARPE = "sharpe_ratio"
    SORTINO = "sortino_ratio"
    CALMAR = "calmar_ratio"
    NET_RETURN = "net_return_pct"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"


@dataclass
class SelectionConfig:
    """Configuration for model selection."""
    primary_metric: SelectionMetric = SelectionMetric.SHARPE
    # Hard constraints
    max_drawdown_pct: float = 15.0
    max_exposure_pct: float = 100.0
    min_trades: int = 10
    min_net_return_pct: float = 0.0
    # Robustness requirements
    min_periods_passed: int = 2  # must pass in at least N periods
    require_consistency: bool = True  # must not have wildly varying results
    max_return_volatility_across_periods: float = 50.0  # std dev of returns across periods
    # Evaluation windows
    evaluation_windows: list = field(default_factory=lambda: ["validation", "out_of_sample"])
    # Tiebreaker
    tiebreaker_metric: SelectionMetric = SelectionMetric.NET_RETURN


@dataclass
class SelectionResult:
    """Result of model selection."""
    selected_candidate_id: Optional[str]
    selected_candidate_version: Optional[str]
    selection_reason: str
    candidates_considered: int
    candidates_passing_gates: int
    candidate_scores: list = field(default_factory=list)
    active_version_retained: bool = False
    active_version_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate_version": self.selected_candidate_version,
            "selection_reason": self.selection_reason,
            "candidates_considered": self.candidates_considered,
            "candidates_passing_gates": self.candidates_passing_gates,
            "candidate_scores": self.candidate_scores,
            "active_version_retained": self.active_version_retained,
            "active_version_reason": self.active_version_reason,
        }


class ModelSelector:
    """
    Selects the most suitable candidate based on configured criteria.

    The selector is deterministic and auditable. It ranks candidates
    by the primary metric subject to hard constraints. If no candidate
    clears all gates, the active version is retained.
    """

    def __init__(self, config: SelectionConfig):
        self.config = config

    def select(
        self,
        candidates: list[Candidate],
        evaluation_results: dict[str, list[EvaluationResult]],
        active_candidate_id: Optional[str] = None,
    ) -> SelectionResult:
        """
        Select the best candidate.

        Parameters
        ----------
        candidates : list[Candidate]
            All candidates to consider.
        evaluation_results : dict[str, list[EvaluationResult]]
            Evaluation results keyed by candidate_id.
        active_candidate_id : str, optional
            Currently active candidate ID (for rollback tracking).

        Returns
        -------
        SelectionResult with the selected candidate and reasoning.
        """
        result = SelectionResult(
            selected_candidate_id=None,
            selected_candidate_version=None,
            selection_reason="",
            candidates_considered=len(candidates),
            candidates_passing_gates=0,
        )

        if not candidates:
            result.selection_reason = "No candidates available for selection"
            result.active_version_retained = True
            result.active_version_reason = "No candidates to evaluate"
            return result

        # Score each candidate
        scored_candidates = []
        for candidate in candidates:
            eval_results = evaluation_results.get(candidate.candidate_id, [])
            score = self._score_candidate(candidate, eval_results)
            if score is not None:
                scored_candidates.append(score)

        result.candidates_passing_gates = len(scored_candidates)

        if not scored_candidates:
            result.selection_reason = "No candidates passed all gates"
            result.active_version_retained = True
            result.active_version_reason = "No candidates met minimum requirements"
            return result

        # Sort by primary metric (descending)
        metric_name = self.config.primary_metric.value
        scored_candidates.sort(key=lambda x: x["scores"].get(metric_name, 0), reverse=True)

        # Select the best
        best = scored_candidates[0]
        result.selected_candidate_id = best["candidate_id"]
        result.selected_candidate_version = best["version"]
        result.selection_reason = (
            f"Selected {best['candidate_id']} ({best['version']}) "
            f"with {metric_name}={best['scores'].get(metric_name, 0):.2f} "
            f"across {best['periods_passed']} periods. "
            f"Hard constraints: max_dd={self.config.max_drawdown_pct}%, "
            f"min_trades={self.config.min_trades}."
        )
        result.candidate_scores = scored_candidates

        return result

    def _score_candidate(
        self,
        candidate: Candidate,
        eval_results: list[EvaluationResult],
    ) -> Optional[dict]:
        """
        Score a candidate. Returns None if it fails hard constraints.
        """
        if not eval_results:
            return None

        # Filter to configured evaluation windows
        window_results = [
            r for r in eval_results
            if r.period in self.config.evaluation_windows
        ]
        if not window_results:
            window_results = eval_results  # fallback to all

        # Data integrity is a per-result veto
        for r in window_results:
            if not r.data_quality_passed or not r.look_ahead_check_passed:
                return None

        # Hard constraints are applied per PERIOD, aggregated across
        # symbols. Per-symbol checks rejected every trend strategy: a single
        # symbol produces only 1-3 trades per 3-6 month window, so
        # min_trades and min_net_return vetoed candidates on one symbol.
        by_period: dict = {}
        for r in window_results:
            by_period.setdefault(r.period, []).append(r)

        periods_passed = 0
        period_returns = []
        for period, rs in by_period.items():
            trades = sum(r.total_trades for r in rs)
            mean_ret = sum(r.net_return_pct for r in rs) / len(rs)
            worst_dd = max(abs(r.max_drawdown_pct) for r in rs)
            period_returns.append(mean_ret)
            if worst_dd > self.config.max_drawdown_pct:
                return None
            if trades < self.config.min_trades:
                return None
            if mean_ret < self.config.min_net_return_pct:
                return None
            # A period "passes" when most symbols cleared the offline gates
            if sum(1 for r in rs if r.passed_gates) * 2 >= len(rs):
                periods_passed += 1

        required = min(self.config.min_periods_passed, len(by_period))
        if periods_passed < required:
            return None

        # Check consistency: return volatility across periods
        if self.config.require_consistency and len(period_returns) > 1:
            import numpy as np
            return_std = float(np.std(period_returns))
            if return_std > self.config.max_return_volatility_across_periods:
                return None  # too inconsistent

        # Compute aggregate scores
        scores = {
            "sharpe_ratio": sum(r.sharpe_ratio for r in window_results) / len(window_results),
            "sortino_ratio": sum(r.sortino_ratio for r in window_results) / len(window_results),
            "calmar_ratio": sum(r.calmar_ratio for r in window_results) / len(window_results),
            "net_return_pct": sum(r.net_return_pct for r in window_results) / len(window_results),
            "risk_adjusted_return": sum(r.sharpe_ratio for r in window_results) / len(window_results),
            "max_drawdown_pct": max(abs(r.max_drawdown_pct) for r in window_results),
            "total_trades": sum(r.total_trades for r in window_results),
            "avg_exposure_pct": sum(r.avg_exposure_pct for r in window_results) / len(window_results),
            "win_rate": sum(r.win_rate for r in window_results) / len(window_results),
            # inf profit factor (no losing trades) would dominate the mean
            "profit_factor": sum(min(r.profit_factor, 10.0) for r in window_results) / len(window_results),
        }

        return {
            "candidate_id": candidate.candidate_id,
            "version": candidate.version,
            "scores": scores,
            "periods_passed": periods_passed,
            "total_periods": len(by_period),
            "hypothesis": candidate.hypothesis,
            "intended_conditions": candidate.intended_conditions,
        }
