"""
Challenger comparison — compares candidate strategies against the active
strategy and baselines on identical chronological data.

Every attempted variant is recorded. No automatic promotion.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .candidates import Candidate, CandidateStatus
from .evaluation import WalkForwardEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class ComparisonConfig:
    """Configuration for challenger comparison."""
    # Data split (must match the active strategy's evaluation)
    train_days: int = 252
    validation_days: int = 126
    out_of_sample_days: int = 63
    # Cost model (must match the active strategy's evaluation)
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    slippage_pct: float = 0.001
    spread_pct: float = 0.0005
    # Minimum trade frequency (horizon-aware)
    min_trade_frequency: float = 0.011  # ~4 trades/year


@dataclass
class ComparisonResult:
    """Result of comparing a challenger against the active strategy and baselines."""
    challenger_id: str
    challenger_version: str
    active_version: str
    symbol: str
    period: str  # training | validation | out_of_sample

    # Challenger metrics
    challenger_return_pct: float = 0.0
    challenger_sharpe: float = 0.0
    challenger_max_drawdown_pct: float = 0.0
    challenger_trades: int = 0
    challenger_profit_factor: float = 0.0

    # Active strategy metrics
    active_return_pct: float = 0.0
    active_sharpe: float = 0.0
    active_max_drawdown_pct: float = 0.0
    active_trades: int = 0
    active_profit_factor: float = 0.0

    # Baseline metrics
    baseline_sma_return_pct: float = 0.0
    baseline_sma_sharpe: float = 0.0
    baseline_buyhold_return_pct: float = 0.0
    baseline_buyhold_sharpe: float = 0.0

    # Comparison verdict
    challenger_beats_active: bool = False
    challenger_beats_baseline: bool = False
    challenger_passes_gates: bool = False
    rejection_reasons: list = field(default_factory=list)

    # Metadata
    compared_at: str = ""
    data_start: str = ""
    data_end: str = ""

    def __post_init__(self):
        if not self.compared_at:
            self.compared_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "challenger_id": self.challenger_id,
            "challenger_version": self.challenger_version,
            "active_version": self.active_version,
            "symbol": self.symbol,
            "period": self.period,
            "challenger_return_pct": round(self.challenger_return_pct, 2),
            "challenger_sharpe": round(self.challenger_sharpe, 2),
            "challenger_max_drawdown_pct": round(self.challenger_max_drawdown_pct, 2),
            "challenger_trades": self.challenger_trades,
            "challenger_profit_factor": round(self.challenger_profit_factor, 2),
            "active_return_pct": round(self.active_return_pct, 2),
            "active_sharpe": round(self.active_sharpe, 2),
            "active_max_drawdown_pct": round(self.active_max_drawdown_pct, 2),
            "active_trades": self.active_trades,
            "active_profit_factor": round(self.active_profit_factor, 2),
            "baseline_sma_return_pct": round(self.baseline_sma_return_pct, 2),
            "baseline_sma_sharpe": round(self.baseline_sma_sharpe, 2),
            "baseline_buyhold_return_pct": round(self.baseline_buyhold_return_pct, 2),
            "baseline_buyhold_sharpe": round(self.baseline_buyhold_sharpe, 2),
            "challenger_beats_active": self.challenger_beats_active,
            "challenger_beats_baseline": self.challenger_beats_baseline,
            "challenger_passes_gates": self.challenger_passes_gates,
            "rejection_reasons": self.rejection_reasons,
            "compared_at": self.compared_at,
            "data_start": self.data_start,
            "data_end": self.data_end,
        }


class ChallengerComparison:
    """
    Compares challenger strategies against the active strategy and baselines
    on identical chronological data splits.

    Uses the same WalkForwardEvaluator and the same train/validation/out_of_sample
    splits for all strategies. No look-ahead: each strategy only sees data
    available at that point in time.
    """

    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.evaluator = WalkForwardEvaluator(
            commission_per_share=config.commission_per_share,
            min_commission=config.min_commission,
            slippage_pct=config.slippage_pct,
            spread_pct=config.spread_pct,
            min_trade_frequency=config.min_trade_frequency,
        )

    def compare(
        self,
        challenger: Candidate,
        challenger_strategy,
        active_strategy,
        df: pd.DataFrame,
        symbol: str,
        period: str,
    ) -> ComparisonResult:
        """
        Compare a challenger against the active strategy on identical data.

        Parameters
        ----------
        challenger : Candidate
            The challenger candidate.
        challenger_strategy : object
            Strategy instance with backtest() method.
        active_strategy : object
            Active strategy instance with backtest() method.
        df : pd.DataFrame
            Historical OHLCV data (chronologically ordered).
        symbol : str
            Symbol being compared.
        period : str
            Evaluation period label.

        Returns
        -------
        ComparisonResult with full comparison metrics.
        """
        result = ComparisonResult(
            challenger_id=challenger.candidate_id,
            challenger_version=challenger.version,
            active_version=getattr(active_strategy, "version", "unknown"),
            symbol=symbol,
            period=period,
        )

        if df.empty:
            result.rejection_reasons.append("empty_dataframe")
            return result

        # Evaluate challenger
        try:
            challenger_eval = self.evaluator.evaluate(
                candidate_id=challenger.candidate_id,
                strategy=challenger_strategy,
                df=df,
                symbol=symbol,
                period=period,
            )
            result.challenger_return_pct = challenger_eval.net_return_pct
            result.challenger_sharpe = challenger_eval.sharpe_ratio
            result.challenger_max_drawdown_pct = challenger_eval.max_drawdown_pct
            result.challenger_trades = challenger_eval.total_trades
            result.challenger_profit_factor = challenger_eval.profit_factor
        except Exception as e:
            result.rejection_reasons.append(f"challenger_eval_error: {e}")

        # Evaluate active strategy on the SAME data
        try:
            active_eval = self.evaluator.evaluate(
                candidate_id="active",
                strategy=active_strategy,
                df=df,
                symbol=symbol,
                period=period,
            )
            result.active_return_pct = active_eval.net_return_pct
            result.active_sharpe = active_eval.sharpe_ratio
            result.active_max_drawdown_pct = active_eval.max_drawdown_pct
            result.active_trades = active_eval.total_trades
            result.active_profit_factor = active_eval.profit_factor
        except Exception as e:
            result.rejection_reasons.append(f"active_eval_error: {e}")

        # Baselines (from challenger eval — they're computed on the same data)
        try:
            result.baseline_sma_return_pct = challenger_eval.baseline_sma_return_pct
            result.baseline_sma_sharpe = challenger_eval.baseline_sma_sharpe
            result.baseline_buyhold_return_pct = challenger_eval.baseline_buyhold_return_pct
            result.baseline_buyhold_sharpe = challenger_eval.baseline_buyhold_sharpe
        except Exception:
            pass

        # Comparison verdicts
        result.challenger_beats_active = (
            result.challenger_sharpe > result.active_sharpe
        )
        result.challenger_beats_baseline = (
            result.challenger_sharpe > max(result.baseline_sma_sharpe, result.baseline_buyhold_sharpe)
        )

        # Gate assessment
        if result.challenger_trades < 4:  # min_trades_per_year
            result.rejection_reasons.append(
                f"Insufficient trades: {result.challenger_trades} < 4"
            )
        if result.challenger_profit_factor < 1.0:
            result.rejection_reasons.append(
                f"Profit factor {result.challenger_profit_factor:.2f} < 1.0"
            )
        if not result.challenger_beats_baseline:
            result.rejection_reasons.append(
                f"Does not beat any baseline on Sharpe "
                f"(challenger={result.challenger_sharpe:.2f}, "
                f"SMA={result.baseline_sma_sharpe:.2f}, "
                f"B&H={result.baseline_buyhold_sharpe:.2f})"
            )

        result.challenger_passes_gates = len(result.rejection_reasons) == 0
        result.data_start = str(df.index[0]) if len(df) > 0 else ""
        result.data_end = str(df.index[-1]) if len(df) > 0 else ""

        return result

    def compare_all_periods(
        self,
        challenger: Candidate,
        challenger_strategy,
        active_strategy,
        df: pd.DataFrame,
        symbol: str,
    ) -> list[ComparisonResult]:
        """
        Compare challenger across all periods (training, validation, out_of_sample).

        Uses identical data splits for all strategies.
        """
        results = []
        total_days = self.config.train_days + self.config.validation_days + self.config.out_of_sample_days

        if len(df) < total_days:
            # Use what we have, proportionally
            train_end = int(len(df) * 0.5)
            val_end = int(len(df) * 0.75)
        else:
            train_end = self.config.train_days
            val_end = self.config.train_days + self.config.validation_days

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        oos_df = df.iloc[val_end:]

        for period_name, period_df in [
            ("training", train_df),
            ("validation", val_df),
            ("out_of_sample", oos_df),
        ]:
            if len(period_df) < 60:
                continue

            result = self.compare(
                challenger=challenger,
                challenger_strategy=challenger_strategy,
                active_strategy=active_strategy,
                df=period_df,
                symbol=symbol,
                period=period_name,
            )
            results.append(result)

        return results
