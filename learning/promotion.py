"""
Promotion gates for candidate strategies.

A candidate that passes offline evaluation may enter paper trading
only as a separately identified version. Promotion to active requires:
  - Minimum paper-trading duration
  - Minimum trade/sample count
  - Comparison against active strategy
  - Explicit operator approval
  - Previous version available for rollback

NO automatic promotion. NO automatic live trading. NO risk limit changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from enum import Enum

from .candidates import Candidate, CandidateStatus

logger = logging.getLogger(__name__)


class PromotionDecision(str, Enum):
    PENDING = "pending"
    READY_FOR_PAPER = "ready_for_paper"
    READY_FOR_PROMOTION = "ready_for_promotion"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_MORE_DATA = "needs_more_data"


@dataclass
class PromotionGateConfig:
    """Configuration for promotion gates."""
    # Paper trading requirements
    min_paper_days: int = 30
    min_paper_trades: int = 10
    min_paper_symbols: int = 2

    # Performance requirements
    min_paper_sharpe: float = 0.5
    max_paper_drawdown_pct: float = 10.0
    min_paper_net_return_pct: float = 0.0

    # Comparison requirements
    must_beat_active_on_sharpe: bool = True
    must_beat_active_on_net_return: bool = False  # softer requirement
    comparison_min_overlap_days: int = 20

    # Safety
    require_operator_approval: bool = True
    allow_auto_promotion: bool = False  # NEVER true in this implementation
    keep_previous_version_for_rollback: bool = True


@dataclass
class PromotionAssessment:
    """Result of evaluating a candidate against promotion gates."""
    candidate_id: str
    decision: PromotionDecision
    paper_days: int = 0
    paper_trades: int = 0
    paper_symbols: int = 0
    paper_sharpe: float = 0.0
    paper_net_return_pct: float = 0.0
    paper_max_drawdown_pct: float = 0.0
    active_sharpe: float = 0.0
    active_net_return_pct: float = 0.0
    gates_passed: list = field(default_factory=list)
    gates_failed: list = field(default_factory=list)
    recommendation: str = ""
    assessed_at: str = ""

    def __post_init__(self):
        if not self.assessed_at:
            self.assessed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "decision": self.decision.value,
            "paper_days": self.paper_days,
            "paper_trades": self.paper_trades,
            "paper_symbols": self.paper_symbols,
            "paper_sharpe": self.paper_sharpe,
            "paper_net_return_pct": self.paper_net_return_pct,
            "paper_max_drawdown_pct": self.paper_max_drawdown_pct,
            "active_sharpe": self.active_sharpe,
            "active_net_return_pct": self.active_net_return_pct,
            "gates_passed": self.gates_passed,
            "gates_failed": self.gates_failed,
            "recommendation": self.recommendation,
            "assessed_at": self.assessed_at,
        }


class PromotionGate:
    """
    Evaluates whether a candidate is ready for paper trading or promotion.

    This gate is deterministic and auditable. It does NOT promote
    candidates — it only assesses readiness. Promotion requires
    explicit operator approval.
    """

    def __init__(self, config: PromotionGateConfig):
        self.config = config

    def assess_for_paper(
        self,
        candidate: Candidate,
        evaluation_results: list,
    ) -> PromotionAssessment:
        """
        Assess whether a candidate is ready to enter paper trading.

        Parameters
        ----------
        candidate : Candidate
            The candidate being assessed.
        evaluation_results : list[EvaluationResult]
            Offline evaluation results across symbols and periods.

        Returns
        -------
        PromotionAssessment with decision and gate details.
        """
        assessment = PromotionAssessment(
            candidate_id=candidate.candidate_id,
            decision=PromotionDecision.PENDING,
        )

        if not evaluation_results:
            assessment.decision = PromotionDecision.NEEDS_MORE_DATA
            assessment.recommendation = "No evaluation results available"
            return assessment

        # Check that candidate passed offline gates
        passed_offline = [r for r in evaluation_results if r.passed_gates]
        if not passed_offline:
            assessment.decision = PromotionDecision.REJECTED
            assessment.gates_failed.append("No evaluation results passed offline gates")
            assessment.recommendation = "Candidate failed offline evaluation"
            return assessment

        # Check data quality across all evaluations
        all_data_quality = all(r.data_quality_passed for r in evaluation_results)
        if not all_data_quality:
            assessment.gates_failed.append("Data quality issues in some evaluations")

        # Check trade count across all evaluations
        total_trades = sum(r.total_trades for r in evaluation_results)
        if total_trades < self.config.min_paper_trades:
            assessment.gates_failed.append(
                f"Total trades {total_trades} < minimum {self.config.min_paper_trades}"
            )

        # Check that candidate beats baselines on average
        avg_candidate_sharpe = sum(r.sharpe_ratio for r in evaluation_results) / len(evaluation_results)
        avg_sma_sharpe = sum(r.baseline_sma_sharpe for r in evaluation_results) / len(evaluation_results)
        avg_bh_sharpe = sum(r.baseline_buyhold_sharpe for r in evaluation_results) / len(evaluation_results)

        if avg_candidate_sharpe <= max(avg_sma_sharpe, avg_bh_sharpe):
            assessment.gates_failed.append(
                f"Avg Sharpe {avg_candidate_sharpe:.2f} does not beat "
                f"best baseline {max(avg_sma_sharpe, avg_bh_sharpe):.2f}"
            )

        # Check parameter bounds
        bound_errors = candidate.config.validate_bounds()
        if bound_errors:
            assessment.gates_failed.extend(bound_errors)

        if assessment.gates_failed:
            assessment.decision = PromotionDecision.REJECTED
            assessment.recommendation = "Failed offline gates: " + "; ".join(assessment.gates_failed[:3])
        else:
            assessment.decision = PromotionDecision.READY_FOR_PAPER
            assessment.gates_passed.append("All offline evaluation gates passed")
            assessment.recommendation = "Ready for paper trading as separately identified version"

        return assessment

    def assess_for_promotion(
        self,
        candidate: Candidate,
        paper_results: dict,
        active_results: dict,
    ) -> PromotionAssessment:
        """
        Assess whether a paper-trading candidate is ready for promotion.

        Parameters
        ----------
        candidate : Candidate
            The candidate in paper trading.
        paper_results : dict
            Paper trading performance results.
        active_results : dict
            Active strategy performance for comparison.

        Returns
        -------
        PromotionAssessment with decision and gate details.
        """
        assessment = PromotionAssessment(
            candidate_id=candidate.candidate_id,
            decision=PromotionDecision.PENDING,
        )

        # --- Paper trading duration gate ---
        assessment.paper_days = paper_results.get("days_in_paper", 0)
        if assessment.paper_days < self.config.min_paper_days:
            assessment.gates_failed.append(
                f"Paper duration {assessment.paper_days}d < minimum {self.config.min_paper_days}d"
            )

        # --- Paper trading trade count gate ---
        assessment.paper_trades = paper_results.get("total_trades", 0)
        if assessment.paper_trades < self.config.min_paper_trades:
            assessment.gates_failed.append(
                f"Paper trades {assessment.paper_trades} < minimum {self.config.min_paper_trades}"
            )

        # --- Paper trading symbol coverage gate ---
        assessment.paper_symbols = paper_results.get("symbols_traded", 0)
        if assessment.paper_symbols < self.config.min_paper_symbols:
            assessment.gates_failed.append(
                f"Paper symbols {assessment.paper_symbols} < minimum {self.config.min_paper_symbols}"
            )

        # --- Paper performance gates ---
        assessment.paper_sharpe = paper_results.get("sharpe_ratio", 0.0)
        if assessment.paper_sharpe < self.config.min_paper_sharpe:
            assessment.gates_failed.append(
                f"Paper Sharpe {assessment.paper_sharpe:.2f} < minimum {self.config.min_paper_sharpe}"
            )

        assessment.paper_max_drawdown_pct = paper_results.get("max_drawdown_pct", 0.0)
        if assessment.paper_max_drawdown_pct > self.config.max_paper_drawdown_pct:
            assessment.gates_failed.append(
                f"Paper max drawdown {assessment.paper_max_drawdown_pct:.2f}% > maximum {self.config.max_paper_drawdown_pct}%"
            )

        assessment.paper_net_return_pct = paper_results.get("net_return_pct", 0.0)
        if assessment.paper_net_return_pct < self.config.min_paper_net_return_pct:
            assessment.gates_failed.append(
                f"Paper net return {assessment.paper_net_return_pct:.2f}% < minimum {self.config.min_paper_net_return_pct}%"
            )

        # --- Comparison against active strategy ---
        assessment.active_sharpe = active_results.get("sharpe_ratio", 0.0)
        assessment.active_net_return_pct = active_results.get("net_return_pct", 0.0)

        if self.config.must_beat_active_on_sharpe:
            if assessment.paper_sharpe <= assessment.active_sharpe:
                assessment.gates_failed.append(
                    f"Paper Sharpe {assessment.paper_sharpe:.2f} <= active {assessment.active_sharpe:.2f}"
                )

        if self.config.must_beat_active_on_net_return:
            if assessment.paper_net_return_pct <= assessment.active_net_return_pct:
                assessment.gates_failed.append(
                    f"Paper return {assessment.paper_net_return_pct:.2f}% <= active {assessment.active_net_return_pct:.2f}%"
                )

        # --- Decision ---
        if assessment.gates_failed:
            assessment.decision = PromotionDecision.NEEDS_MORE_DATA
            assessment.recommendation = "Not ready: " + "; ".join(assessment.gates_failed[:3])
        elif self.config.require_operator_approval:
            assessment.decision = PromotionDecision.READY_FOR_PROMOTION
            assessment.gates_passed.append("All paper trading gates passed")
            assessment.recommendation = (
                "Ready for operator review. Promotion requires explicit operator approval. "
                "Previous version will be preserved for rollback."
            )
        elif self.config.allow_auto_promotion:
            # This path exists but is NEVER enabled in this implementation
            assessment.decision = PromotionDecision.APPROVED
            assessment.recommendation = "Auto-promotion enabled (NOT RECOMMENDED)"
        else:
            assessment.decision = PromotionDecision.READY_FOR_PROMOTION
            assessment.recommendation = "All gates passed. Awaiting operator approval."

        return assessment
