"""
Baseline strategy: Simple Moving Average (SMA) Crossover.

A common starting point for testing the Alpaca + yfinance integration.
- Buy when short SMA crosses above long SMA
- Sell when short SMA crosses below long SMA

This is the v1.0 baseline — no fancy signals. Keep it simple to validate
the data pipeline before adding complexity.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SMACrossStrategy:
    """
    SMA Crossover Strategy.

    Parameters
    ----------
    sma_short : int
        Short SMA window (e.g. 20 days).
    sma_long : int
        Long SMA window (e.g. 50 days).
    """

    def __init__(self, sma_short: int = 20, sma_long: int = 50):
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.name = f"SMA Cross ({sma_short}/{sma_long})"
        self.version = "v1.0"

    def generate_signal(self, df: pd.DataFrame) -> int:
        """
        Generate a position signal from the latest data.
        Returns: 1 (long), -1 (short/exit), or 0 (hold).
        """
        if len(df) < self.sma_long:
            return 0

        df = df.copy()
        df["sma_short"] = df["Close"].rolling(window=self.sma_short).mean()
        df["sma_long"] = df["Close"].rolling(window=self.sma_long).mean()

        # Position signal: 1 if short > long, 0 otherwise
        df["signal"] = np.where(df["sma_short"] > df["sma_long"], 1, 0)
        # Detect crossover by diffing the signal
        df["position"] = df["signal"].diff()

        latest = df.iloc[-1]
        if pd.isna(latest["position"]):
            return 0

        if latest["position"] == 1:
            return 1   # Buy signal
        elif latest["position"] == -1:
            return -1  # Sell signal
        else:
            # Hold previous position
            return int(latest["signal"])

    def backtest(self, df: pd.DataFrame) -> dict:
        """
        Run a backtest on historical OHLCV data.
        Returns performance metrics.
        """
        if len(df) < self.sma_long:
            return {
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "total_trades": 0,
                "error": "insufficient data",
            }

        df = df.copy()
        df["sma_short"] = df["Close"].rolling(window=self.sma_short).mean()
        df["sma_long"] = df["Close"].rolling(window=self.sma_long).mean()
        df["signal"] = np.where(df["sma_short"] > df["sma_long"], 1, 0)
        df["position"] = df["signal"].diff()

        # Track trades
        positions = df["signal"].ffill().fillna(0)
        daily_returns = df["Close"].pct_change()

        # Strategy returns (position lagged by 1 day to avoid lookahead bias)
        strategy_returns = (positions.shift(1) * daily_returns).fillna(0)

        # Cumulative returns
        cum_stock = (1 + daily_returns).cumprod()
        cum_strategy = (1 + strategy_returns).cumprod()

        total_return_pct = (cum_strategy.iloc[-1] - 1) * 100

        # Max drawdown
        rolling_peak = cum_strategy.cummax()
        drawdown = (cum_strategy - rolling_peak) / rolling_peak
        max_drawdown_pct = drawdown.min() * 100

        # Sharpe ratio (annualized)
        if strategy_returns.std() > 0:
            sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Count trades (crossover events: position changes by 1)
        total_trades = int((df["position"].abs() == 1).sum())

        # Compute per-trade returns for profit factor
        trade_returns = []
        in_position = False
        entry_price = 0.0
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            if not in_position and prev["position"] == 1:
                in_position = True
                entry_price = row["Close"]
            elif in_position and prev["position"] == -1:
                exit_price = row["Close"]
                trade_returns.append((exit_price - entry_price) / entry_price * 100)
                in_position = False
        if in_position:
            exit_price = df.iloc[-1]["Close"]
            trade_returns.append((exit_price - entry_price) / entry_price * 100)

        return {
            "total_return_pct": round(total_return_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "total_trades": total_trades,
            "final_strategy_value": round(cum_strategy.iloc[-1], 4),
            "final_buyhold_value": round(cum_stock.iloc[-1], 4),
            "trade_returns": trade_returns,
            "daily_returns": strategy_returns,
        }
