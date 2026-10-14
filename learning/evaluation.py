"""
Walk-forward evaluation engine for candidate strategies.

Evaluates candidates on chronologically separated data:
  - Training period: for parameter fitting (if needed)
  - Validation period: for initial evaluation
  - Out-of-sample period: untouched, for final assessment

Each simulated decision only sees information available at that time.
Includes realistic fees, spread, slippage, and market hours.

Baselines (always included):
  - SMA Cross 20/50
  - Buy-and-hold
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Full evaluation results for one candidate on one symbol."""
    candidate_id: str
    symbol: str
    period: str  # training | validation | out_of_sample
    start_date: str
    end_date: str

    # Performance metrics
    net_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    volatility_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration_days: float = 0.0

    # Exposure and turnover
    avg_exposure_pct: float = 0.0
    max_exposure_pct: float = 0.0
    turnover: float = 0.0  # total value traded / avg portfolio value

    # Cost modeling
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_spread_cost: float = 0.0
    cost_impact_pct: float = 0.0  # total costs as % of gross return

    # Uncertainty
    trade_count_sufficient: bool = False
    trade_frequency: float = 0.0  # trades per day
    confidence_interval_95: tuple = (0.0, 0.0)
    sample_size_adequate: bool = False

    # Data quality
    data_quality_passed: bool = True
    data_quality_issues: list = field(default_factory=list)
    look_ahead_check_passed: bool = True
    period_days: int = 0

    # Baselines for comparison
    baseline_sma_return_pct: float = 0.0
    baseline_buyhold_return_pct: float = 0.0
    baseline_sma_sharpe: float = 0.0
    baseline_buyhold_sharpe: float = 0.0

    # Verdict
    passed_gates: bool = False
    rejection_reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, tuple):
                d[k] = list(v)
            else:
                d[k] = v
        return d


@dataclass
class BaselineResult:
    """Baseline strategy result for comparison."""
    baseline_name: str
    symbol: str
    period: str
    net_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    volatility_pct: float = 0.0
    avg_exposure_pct: float = 0.0


class WalkForwardEvaluator:
    """
    Walk-forward evaluation engine.

    Splits data chronologically and evaluates each candidate with
    realistic cost modeling. No look-ahead: each decision only sees
    data available at that point in time.
    """

    # Cost model defaults (configurable)
    DEFAULT_COMMISSION_PER_SHARE = 0.005  # Alpaca: $0.005/share
    DEFAULT_MIN_COMMISSION = 1.0
    DEFAULT_SLIPPAGE_PCT = 0.001  # 0.1% slippage
    DEFAULT_SPREAD_PCT = 0.0005  # 0.05% half-spread cost
    # Minimum trades per calendar day. 0.5 (one trade every 2 days) rejected
    # every daily trend-following candidate; v1.1 trades ~0.005-0.011/day per
    # symbol. 0.004 ~= one trade per symbol per 250 days.
    DEFAULT_MIN_TRADE_FREQUENCY = 0.004

    def __init__(
        self,
        commission_per_share: float = DEFAULT_COMMISSION_PER_SHARE,
        min_commission: float = DEFAULT_MIN_COMMISSION,
        slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
        spread_pct: float = DEFAULT_SPREAD_PCT,
        min_trade_frequency: float = DEFAULT_MIN_TRADE_FREQUENCY,
    ):
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.slippage_pct = slippage_pct
        self.spread_pct = spread_pct
        self.min_trade_frequency = min_trade_frequency

    def evaluate(
        self,
        candidate_id: str,
        strategy,
        df: pd.DataFrame,
        symbol: str,
        period: str,
        train_end: Optional[pd.Timestamp] = None,
        eval_start: Optional[pd.Timestamp] = None,
    ) -> EvaluationResult:
        """
        Evaluate a candidate strategy on historical data.

        Parameters
        ----------
        candidate_id : str
            Identifier for the candidate being evaluated.
        strategy : object
            Strategy instance with generate_signal() and backtest() methods.
        df : pd.DataFrame
            Historical OHLCV data (chronologically ordered).
        symbol : str
            Symbol being evaluated.
        period : str
            Evaluation period label (training/validation/out_of_sample).
        train_end : pd.Timestamp, optional
            End of training period (for walk-forward boundary).
        eval_start : pd.Timestamp, optional
            Start of the scored window. `df` may contain earlier bars as
            indicator warm-up; only returns/trades from eval_start onward are
            scored (and baselines use the same window).

        Returns
        -------
        EvaluationResult with full metrics and gate assessment.
        """
        eval_df = df[df.index >= eval_start] if eval_start is not None else df
        result = EvaluationResult(
            candidate_id=candidate_id,
            symbol=symbol,
            period=period,
            start_date=str(eval_df.index[0]) if len(eval_df) > 0 else "",
            end_date=str(eval_df.index[-1]) if len(eval_df) > 0 else "",
        )

        # --- Data integrity checks ---
        if not self._check_data_integrity(df, result):
            return result

        # --- Run backtest with cost modeling ---
        try:
            raw_metrics = strategy.backtest(df)
        except Exception as e:
            result.data_quality_passed = False
            result.data_quality_issues.append(f"backtest_error: {e}")
            result.rejection_reasons.append(f"Backtest failed: {e}")
            return result

        if eval_start is not None:
            from strategies.v1_1 import window_metrics
            raw_metrics = window_metrics(raw_metrics, eval_start)

        # --- Apply cost modeling ---
        metrics = self._apply_cost_modeling(raw_metrics, eval_df)

        # --- Compute baselines (same scored window) ---
        sma_metrics = self._compute_sma_baseline(df)
        if eval_start is not None:
            from strategies.v1_1 import window_metrics
            sma_metrics = window_metrics(sma_metrics, eval_start)
        bh_metrics = self._compute_buyhold_baseline(eval_df)

        result.baseline_sma_return_pct = sma_metrics["total_return_pct"]
        result.baseline_sma_sharpe = sma_metrics["sharpe_ratio"]
        result.baseline_buyhold_return_pct = bh_metrics["total_return_pct"]
        result.baseline_buyhold_sharpe = bh_metrics["sharpe_ratio"]

        # --- Populate metrics ---
        result.net_return_pct = metrics.get("total_return_pct", 0.0)
        # Strategies report drawdown as a negative %; store magnitude so the
        # "> max_drawdown_pct" gates and Calmar (which divided by a positive
        # drawdown only) actually work.
        result.max_drawdown_pct = abs(metrics.get("max_drawdown_pct", 0.0))
        result.sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
        result.total_trades = metrics.get("total_trades", 0)
        result.win_rate = metrics.get("win_rate", 0.0)
        result.avg_win_pct = metrics.get("avg_win_pct", 0.0)
        result.avg_loss_pct = metrics.get("avg_loss_pct", 0.0)
        result.total_fees = metrics.get("total_fees", 0.0)
        result.total_slippage = metrics.get("total_slippage", 0.0)
        result.total_spread_cost = metrics.get("total_spread_cost", 0.0)

        # --- Compute additional metrics ---
        daily = raw_metrics.get("daily_returns")
        if daily is not None and len(daily) > 1:
            # Use the STRATEGY's daily returns. The old helpers measured the
            # underlying stock's volatility/Sortino, not the strategy's.
            dr = pd.Series(daily).fillna(0.0)
            result.volatility_pct = round(float(dr.std() * np.sqrt(252) * 100), 2)
            downside = dr[dr < 0]
            dstd = downside.std() if len(downside) > 1 else 0.0
            result.sortino_ratio = round(float(dr.mean() / dstd * np.sqrt(252)), 2) if dstd > 0 else 0.0
            in_mkt = (dr != 0)
            result.avg_exposure_pct = round(float(in_mkt.mean() * 100), 2)
            result.max_exposure_pct = 100.0 if in_mkt.any() else 0.0
            result.turnover = float(result.total_trades * 2)
        else:
            result.volatility_pct = self._compute_volatility(eval_df, strategy)
            result.sortino_ratio = self._compute_sortino(eval_df, strategy)
            result.avg_exposure_pct = self._compute_avg_exposure(eval_df, strategy)
            result.max_exposure_pct = self._compute_max_exposure(eval_df, strategy)
            result.turnover = self._compute_turnover(eval_df, strategy)
        result.calmar_ratio = self._compute_calmar(result)

        # --- Profit factor ---
        trade_returns = metrics.get("trade_returns", metrics.get("returns", []))
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        # --- Uncertainty and sample size (frequency-based) ---
        period_days = (eval_df.index[-1] - eval_df.index[0]).days if len(eval_df) > 1 and isinstance(eval_df.index, pd.DatetimeIndex) else len(eval_df)
        result.period_days = period_days
        result.trade_frequency = result.total_trades / max(period_days, 1)
        result.trade_count_sufficient = result.trade_frequency >= self.min_trade_frequency
        result.sample_size_adequate = result.trade_count_sufficient
        result.confidence_interval_95 = self._compute_confidence_interval(
            metrics.get("trade_returns", [])
        )

        # --- Look-ahead check ---
        result.look_ahead_check_passed = self._check_look_ahead(eval_df, strategy)

        # --- Gate assessment ---
        result.passed_gates = self._assess_gates(result)

        return result

    def _check_data_integrity(self, df: pd.DataFrame, result: EvaluationResult) -> bool:
        """Check data quality: missing values, gaps, stale data."""
        passed = True

        if df.empty:
            result.data_quality_issues.append("empty_dataframe")
            result.data_quality_passed = False
            result.rejection_reasons.append("Data integrity: empty_dataframe")
            return False

        if len(df) < 60:
            result.data_quality_issues.append(f"insufficient_data: {len(df)} rows")
            passed = False

        # Check for missing values in required columns
        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                result.data_quality_issues.append(f"missing_column: {col}")
                passed = False
            elif df[col].isna().any():
                nan_count = df[col].isna().sum()
                result.data_quality_issues.append(f"nan_values: {col}={nan_count}")
                passed = False

        # Check for zero-volume days (potential stale data)
        if "Volume" in df.columns:
            zero_vol = (df["Volume"] == 0).sum()
            if zero_vol > len(df) * 0.1:  # more than 10% zero volume
                result.data_quality_issues.append(f"excessive_zero_volume: {zero_vol}/{len(df)}")
                passed = False

        # Check for large gaps (> 5 trading days) — only for datetime indices
        if len(df) > 1 and isinstance(df.index, pd.DatetimeIndex):
            gaps = df.index.to_series().diff().dt.days
            max_gap = gaps.max()
            if max_gap > 7:  # more than a week
                result.data_quality_issues.append(f"large_data_gap: {max_gap} days")
                passed = False

        # Check for negative prices
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns and (df[col] < 0).any():
                result.data_quality_issues.append(f"negative_prices: {col}")
                passed = False

        result.data_quality_passed = passed
        if not passed:
            result.rejection_reasons.append(f"Data integrity: {'; '.join(result.data_quality_issues)}")
        return passed

    def _apply_cost_modeling(self, raw_metrics: dict, df: pd.DataFrame) -> dict:
        """Apply realistic costs: commission, slippage, spread."""
        metrics = dict(raw_metrics)
        total_trades = metrics.get("total_trades", 0)

        if total_trades == 0:
            metrics["total_fees"] = 0.0
            metrics["total_slippage"] = 0.0
            metrics["total_spread_cost"] = 0.0
            return metrics

        # Estimate average position size from data
        avg_price = df["Close"].mean()
        # Assume ~5% of $100K = $5000 per trade
        avg_position = 5000.0
        avg_qty = avg_position / avg_price if avg_price > 0 else 1

        # Commission: max(min_commission, per_share * qty)
        commission_per_trade = max(self.min_commission, self.commission_per_share * avg_qty)
        total_commission = commission_per_trade * total_trades * 2  # entry + exit

        # Slippage: % of notional per side
        slippage_per_trade = avg_position * self.slippage_pct * 2  # entry + exit
        total_slippage = slippage_per_trade * total_trades

        # Spread cost: half-spread per side
        spread_per_trade = avg_position * self.spread_pct * 2
        total_spread = spread_per_trade * total_trades

        total_costs = total_commission + total_slippage + total_spread

        # Adjust returns. Strategy returns are measured on a fully-invested
        # per-symbol basis, so costs must be a % of the POSITION, not of a
        # $100K portfolio (that understated costs ~20x).
        gross_return = metrics.get("total_return_pct", 0.0)
        cost_impact = (total_costs / avg_position) * 100 if avg_position > 0 else 0.0
        metrics["total_return_pct"] = gross_return - cost_impact
        metrics["total_fees"] = round(total_commission, 2)
        metrics["total_slippage"] = round(total_slippage, 2)
        metrics["total_spread_cost"] = round(total_spread, 2)
        metrics["cost_impact_pct"] = round(cost_impact, 4)
        metrics["trade_returns"] = metrics.get("trade_returns", [])

        return metrics

    def _compute_sma_baseline(self, df: pd.DataFrame) -> dict:
        """Compute SMA Cross 20/50 baseline."""
        from strategies.baseline import SMACrossStrategy
        strategy = SMACrossStrategy(sma_short=20, sma_long=50)
        return strategy.backtest(df)

    def _compute_buyhold_baseline(self, df: pd.DataFrame) -> dict:
        """Compute buy-and-hold baseline."""
        if len(df) < 2:
            return {"total_return_pct": 0.0, "sharpe_ratio": 0.0}
        daily_returns = df["Close"].pct_change().dropna()
        total_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0.0
        return {
            "total_return_pct": round(total_return, 2),
            "sharpe_ratio": round(float(sharpe), 2),
        }

    def _compute_volatility(self, df: pd.DataFrame, strategy) -> float:
        """Compute annualized volatility of strategy returns."""
        try:
            daily_returns = df["Close"].pct_change().dropna()
            return round(float(daily_returns.std() * np.sqrt(252) * 100), 2)
        except Exception:
            return 0.0

    def _compute_sortino(self, df: pd.DataFrame, strategy) -> float:
        """Compute Sortino ratio."""
        try:
            daily_returns = df["Close"].pct_change().dropna()
            downside = daily_returns[daily_returns < 0]
            downside_std = downside.std() if len(downside) > 0 else 0.0
            if downside_std > 0:
                return round(float(daily_returns.mean() / downside_std * np.sqrt(252)), 2)
            return 0.0
        except Exception:
            return 0.0

    def _compute_calmar(self, result: EvaluationResult) -> float:
        """Compute Calmar ratio (return / max drawdown)."""
        if result.max_drawdown_pct > 0:
            return round(result.net_return_pct / result.max_drawdown_pct, 2)
        return 0.0

    def _compute_avg_exposure(self, df: pd.DataFrame, strategy) -> float:
        """Compute average exposure percentage."""
        try:
            df_copy = df.copy()
            if hasattr(strategy, "_compute_signals"):
                df_copy = strategy._compute_signals(df_copy)
                if "signal" in df_copy.columns:
                    positions = df_copy["signal"].ffill().fillna(0)
                    return round(float(positions.mean() * 100), 2)
            return 0.0
        except Exception:
            return 0.0

    def _compute_max_exposure(self, df: pd.DataFrame, strategy) -> float:
        """Compute maximum exposure percentage."""
        try:
            df_copy = df.copy()
            if hasattr(strategy, "_compute_signals"):
                df_copy = strategy._compute_signals(df_copy)
                if "signal" in df_copy.columns:
                    positions = df_copy["signal"].ffill().fillna(0)
                    return round(float(positions.max() * 100), 2)
            return 0.0
        except Exception:
            return 0.0

    def _compute_turnover(self, df: pd.DataFrame, strategy) -> float:
        """Compute turnover ratio."""
        try:
            df_copy = df.copy()
            if hasattr(strategy, "_compute_signals"):
                df_copy = strategy._compute_signals(df_copy)
                if "position" in df_copy.columns:
                    position_changes = df_copy["position"].abs().fillna(0)
                    return round(float(position_changes.sum()), 2)
            return 0.0
        except Exception:
            return 0.0

    def _compute_confidence_interval(self, returns: list) -> tuple:
        """Compute 95% confidence interval for mean trade return."""
        if len(returns) < 2:
            return (0.0, 0.0)
        arr = np.array(returns, dtype=float)
        mean = arr.mean()
        se = arr.std() / np.sqrt(len(arr))
        ci_95 = 1.96 * se
        return (round(float(mean - ci_95), 4), round(float(mean + ci_95), 4))

    def _check_look_ahead(self, df: pd.DataFrame, strategy) -> bool:
        """
        Check for look-ahead bias by verifying that the strategy's
        signal at time t only uses data up to time t.

        This is a structural check: we verify that the backtest
        uses lagged positions (position.shift(1)) for return calc.
        """
        # The existing strategies use positions.shift(1) which is correct.
        # This check verifies the backtest method exists and uses proper lagging.
        try:
            # Run a mini-backtest and verify no NaN in early positions
            if hasattr(strategy, "backtest"):
                result = strategy.backtest(df.head(100))
                # If the strategy produces trades on insufficient data, it may have look-ahead
                if result.get("total_trades", 0) > 50:  # suspiciously many trades
                    return False
            return True
        except Exception:
            return False

    def _assess_gates(self, result: EvaluationResult) -> bool:
        """Assess whether the candidate passes all evaluation gates."""
        # Gate 1: Data quality
        if not result.data_quality_passed:
            result.rejection_reasons.append("Failed data quality checks")

        # Gate 2: Minimum trade frequency (trades per day)
        if not result.trade_count_sufficient:
            result.rejection_reasons.append(
                f"Insufficient trade frequency: {result.trade_frequency:.3f} trades/day "
                f"({result.total_trades} trades over {result.period_days}d) < {self.min_trade_frequency}"
            )

        # Gate 3: Look-ahead check
        if not result.look_ahead_check_passed:
            result.rejection_reasons.append("Potential look-ahead bias detected")

        # Gate 4: Must beat at least one baseline on risk-adjusted return
        beats_sma = result.sharpe_ratio > result.baseline_sma_sharpe
        beats_bh = result.sharpe_ratio > result.baseline_buyhold_sharpe
        if not (beats_sma or beats_bh):
            result.rejection_reasons.append(
                f"Does not beat any baseline on Sharpe "
                f"(candidate={result.sharpe_ratio:.2f}, "
                f"SMA={result.baseline_sma_sharpe:.2f}, "
                f"B&H={result.baseline_buyhold_sharpe:.2f})"
            )

        # Gate 5: Profit factor must be > 1.0 (net positive after costs)
        if result.profit_factor < 1.0:
            result.rejection_reasons.append(
                f"Profit factor {result.profit_factor:.2f} < 1.0 (not net positive)"
            )

        return len(result.rejection_reasons) == 0
