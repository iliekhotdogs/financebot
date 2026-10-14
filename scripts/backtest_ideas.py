#!/usr/bin/env python3
"""
Backtest multiple strategy ideas through the walk-forward evaluation framework.

Creates diverse strategy candidates and evaluates them on Alpaca data
with realistic cost modeling. Reports which ideas pass gates and shows
performance vs baselines.

Usage:
    .venv/bin/python scripts/backtest_ideas.py
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BOT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BOT_DIR, ".env"))

from data import DataProvider
from main import resolve_market_universe
from learning.evaluation import WalkForwardEvaluator
from learning.candidates import Candidate, CandidateConfig, CandidateStatus
from learning.candidates import CandidateStore
from learning.audit import LearningAuditLog

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Strategy implementations
# ──────────────────────────────────────────────────────────────

class SMACross:
    """SMA Crossover strategy."""
    def __init__(self, sma_short=20, sma_long=50):
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.name = f"SMA Cross ({sma_short}/{sma_long})"

    def backtest(self, df):
        if len(df) < self.sma_long:
            return {"total_return_pct": 0, "max_drawdown_pct": 0, "sharpe_ratio": 0, "total_trades": 0, "trade_returns": []}
        df = df.copy()
        df["sma_short"] = df["Close"].rolling(self.sma_short).mean()
        df["sma_long"] = df["Close"].rolling(self.sma_long).mean()
        df["signal"] = np.where(df["sma_short"] > df["sma_long"], 1, 0)
        df["position"] = df["signal"].diff()
        positions = df["signal"].ffill().fillna(0)
        daily_returns = df["Close"].pct_change()
        strategy_returns = (positions.shift(1) * daily_returns).fillna(0)
        cum_strategy = (1 + strategy_returns).cumprod()
        total_return = (cum_strategy.iloc[-1] - 1) * 100
        rolling_peak = cum_strategy.cummax()
        drawdown = (cum_strategy - rolling_peak) / rolling_peak
        max_dd = drawdown.min() * 100
        sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0
        total_trades = int((df["position"].abs() == 1).sum())
        # Per-trade returns
        trade_returns = []
        in_pos = False
        entry = 0
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            if not in_pos and prev["position"] == 1:
                in_pos = True
                entry = row["Close"]
            elif in_pos and prev["position"] == -1:
                trade_returns.append((row["Close"] - entry) / entry * 100)
                in_pos = False
        if in_pos:
            trade_returns.append((df.iloc[-1]["Close"] - entry) / entry * 100)
        return {"total_return_pct": round(total_return, 2), "max_drawdown_pct": round(max_dd, 2), "sharpe_ratio": round(float(sharpe), 2), "total_trades": total_trades, "trade_returns": trade_returns}


class RSIMeanReversion:
    """RSI mean reversion: buy oversold, sell overbought."""
    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.name = f"RSI Mean Reversion ({rsi_period}, {oversold}/{overbought})"

    def _compute_rsi(self, df):
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def backtest(self, df):
        if len(df) < self.rsi_period + 1:
            return {"total_return_pct": 0, "max_drawdown_pct": 0, "sharpe_ratio": 0, "total_trades": 0, "trade_returns": []}
        df = df.copy()
        df["rsi"] = self._compute_rsi(df)
        df["signal"] = 0
        df.loc[df["rsi"] < self.oversold, "signal"] = 1
        df.loc[df["rsi"] > self.overbought, "signal"] = 0
        df["position"] = df["signal"].diff()
        positions = df["signal"].ffill().fillna(0)
        daily_returns = df["Close"].pct_change()
        strategy_returns = (positions.shift(1) * daily_returns).fillna(0)
        cum_strategy = (1 + strategy_returns).cumprod()
        total_return = (cum_strategy.iloc[-1] - 1) * 100
        rolling_peak = cum_strategy.cummax()
        drawdown = (cum_strategy - rolling_peak) / rolling_peak
        max_dd = drawdown.min() * 100
        sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0
        total_trades = int((df["position"].abs() == 1).sum())
        trade_returns = []
        in_pos = False
        entry = 0
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            if not in_pos and prev["position"] == 1:
                in_pos = True
                entry = row["Close"]
            elif in_pos and prev["position"] == -1:
                trade_returns.append((row["Close"] - entry) / entry * 100)
                in_pos = False
        if in_pos:
            trade_returns.append((df.iloc[-1]["Close"] - entry) / entry * 100)
        return {"total_return_pct": round(total_return, 2), "max_drawdown_pct": round(max_dd, 2), "sharpe_ratio": round(float(sharpe), 2), "total_trades": total_trades, "trade_returns": trade_returns}


class MACDMomentum:
    """MACD momentum: buy on bullish crossover, sell on bearish."""
    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.name = f"MACD Momentum ({fast}/{slow}/{signal})"

    def backtest(self, df):
        if len(df) < self.slow + self.signal:
            return {"total_return_pct": 0, "max_drawdown_pct": 0, "sharpe_ratio": 0, "total_trades": 0, "trade_returns": []}
        df = df.copy()
        ema_fast = df["Close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        df["signal"] = np.where(macd_line > signal_line, 1, 0)
        df["position"] = df["signal"].diff()
        positions = df["signal"].ffill().fillna(0)
        daily_returns = df["Close"].pct_change()
        strategy_returns = (positions.shift(1) * daily_returns).fillna(0)
        cum_strategy = (1 + strategy_returns).cumprod()
        total_return = (cum_strategy.iloc[-1] - 1) * 100
        rolling_peak = cum_strategy.cummax()
        drawdown = (cum_strategy - rolling_peak) / rolling_peak
        max_dd = drawdown.min() * 100
        sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0
        total_trades = int((df["position"].abs() == 1).sum())
        trade_returns = []
        in_pos = False
        entry = 0
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            if not in_pos and prev["position"] == 1:
                in_pos = True
                entry = row["Close"]
            elif in_pos and prev["position"] == -1:
                trade_returns.append((row["Close"] - entry) / entry * 100)
                in_pos = False
        if in_pos:
            trade_returns.append((df.iloc[-1]["Close"] - entry) / entry * 100)
        return {"total_return_pct": round(total_return, 2), "max_drawdown_pct": round(max_dd, 2), "sharpe_ratio": round(float(sharpe), 2), "total_trades": total_trades, "trade_returns": trade_returns}


class BollingerBands:
    """Bollinger Bands mean reversion."""
    def __init__(self, window=20, num_std=2):
        self.window = window
        self.num_std = num_std
        self.name = f"Bollinger Bands ({window}, {num_std}σ)"

    def backtest(self, df):
        if len(df) < self.window:
            return {"total_return_pct": 0, "max_drawdown_pct": 0, "sharpe_ratio": 0, "total_trades": 0, "trade_returns": []}
        df = df.copy()
        df["sma"] = df["Close"].rolling(self.window).mean()
        df["std"] = df["Close"].rolling(self.window).std()
        df["upper"] = df["sma"] + self.num_std * df["std"]
        df["lower"] = df["sma"] - self.num_std * df["std"]
        df["signal"] = 0
        df.loc[df["Close"] < df["lower"], "signal"] = 1
        df.loc[df["Close"] > df["upper"], "signal"] = 0
        df["position"] = df["signal"].diff()
        positions = df["signal"].ffill().fillna(0)
        daily_returns = df["Close"].pct_change()
        strategy_returns = (positions.shift(1) * daily_returns).fillna(0)
        cum_strategy = (1 + strategy_returns).cumprod()
        total_return = (cum_strategy.iloc[-1] - 1) * 100
        rolling_peak = cum_strategy.cummax()
        drawdown = (cum_strategy - rolling_peak) / rolling_peak
        max_dd = drawdown.min() * 100
        sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0
        total_trades = int((df["position"].abs() == 1).sum())
        trade_returns = []
        in_pos = False
        entry = 0
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            if not in_pos and prev["position"] == 1:
                in_pos = True
                entry = row["Close"]
            elif in_pos and prev["position"] == -1:
                trade_returns.append((row["Close"] - entry) / entry * 100)
                in_pos = False
        if in_pos:
            trade_returns.append((df.iloc[-1]["Close"] - entry) / entry * 100)
        return {"total_return_pct": round(total_return, 2), "max_drawdown_pct": round(max_dd, 2), "sharpe_ratio": round(float(sharpe), 2), "total_trades": total_trades, "trade_returns": trade_returns}


class ZScoreMeanReversion:
    """Z-score mean reversion: buy when price is far below mean."""
    def __init__(self, window=20, entry_z=2.0, exit_z=0.5):
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.name = f"Z-Score Mean Reversion ({window}, ±{entry_z})"

    def backtest(self, df):
        if len(df) < self.window:
            return {"total_return_pct": 0, "max_drawdown_pct": 0, "sharpe_ratio": 0, "total_trades": 0, "trade_returns": []}
        df = df.copy()
        df["sma"] = df["Close"].rolling(self.window).mean()
        df["std"] = df["Close"].rolling(self.window).std()
        df["zscore"] = (df["Close"] - df["sma"]) / df["std"]
        df["signal"] = 0
        df.loc[df["zscore"] < -self.entry_z, "signal"] = 1
        df.loc[df["zscore"] > -self.exit_z, "signal"] = 0
        df["position"] = df["signal"].diff()
        positions = df["signal"].ffill().fillna(0)
        daily_returns = df["Close"].pct_change()
        strategy_returns = (positions.shift(1) * daily_returns).fillna(0)
        cum_strategy = (1 + strategy_returns).cumprod()
        total_return = (cum_strategy.iloc[-1] - 1) * 100
        rolling_peak = cum_strategy.cummax()
        drawdown = (cum_strategy - rolling_peak) / rolling_peak
        max_dd = drawdown.min() * 100
        sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0
        total_trades = int((df["position"].abs() == 1).sum())
        trade_returns = []
        in_pos = False
        entry = 0
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            if not in_pos and prev["position"] == 1:
                in_pos = True
                entry = row["Close"]
            elif in_pos and prev["position"] == -1:
                trade_returns.append((row["Close"] - entry) / entry * 100)
                in_pos = False
        if in_pos:
            trade_returns.append((df.iloc[-1]["Close"] - entry) / entry * 100)
        return {"total_return_pct": round(total_return, 2), "max_drawdown_pct": round(max_dd, 2), "sharpe_ratio": round(float(sharpe), 2), "total_trades": total_trades, "trade_returns": trade_returns}


class DualMomentum:
    """Dual momentum: long if price > SMA, else cash."""
    def __init__(self, sma_window=100):
        self.sma_window = sma_window
        self.name = f"Dual Momentum ({sma_window}d SMA)"

    def backtest(self, df):
        if len(df) < self.sma_window:
            return {"total_return_pct": 0, "max_drawdown_pct": 0, "sharpe_ratio": 0, "total_trades": 0, "trade_returns": []}
        df = df.copy()
        df["sma"] = df["Close"].rolling(self.sma_window).mean()
        df["signal"] = np.where(df["Close"] > df["sma"], 1, 0)
        df["position"] = df["signal"].diff()
        positions = df["signal"].ffill().fillna(0)
        daily_returns = df["Close"].pct_change()
        strategy_returns = (positions.shift(1) * daily_returns).fillna(0)
        cum_strategy = (1 + strategy_returns).cumprod()
        total_return = (cum_strategy.iloc[-1] - 1) * 100
        rolling_peak = cum_strategy.cummax()
        drawdown = (cum_strategy - rolling_peak) / rolling_peak
        max_dd = drawdown.min() * 100
        sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0
        total_trades = int((df["position"].abs() == 1).sum())
        trade_returns = []
        in_pos = False
        entry = 0
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            if not in_pos and prev["position"] == 1:
                in_pos = True
                entry = row["Close"]
            elif in_pos and prev["position"] == -1:
                trade_returns.append((row["Close"] - entry) / entry * 100)
                in_pos = False
        if in_pos:
            trade_returns.append((df.iloc[-1]["Close"] - entry) / entry * 100)
        return {"total_return_pct": round(total_return, 2), "max_drawdown_pct": round(max_dd, 2), "sharpe_ratio": round(float(sharpe), 2), "total_trades": total_trades, "trade_returns": trade_returns}


# ──────────────────────────────────────────────────────────────
# Main backtest runner
# ──────────────────────────────────────────────────────────────

def main():
    import yaml
    config_path = os.path.join(BOT_DIR, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    paper = os.getenv("APCA_PAPER", "true").lower() == "true"
    dp = DataProvider(paper=paper)
    symbols = resolve_market_universe(config, dp)

    evaluator = WalkForwardEvaluator(
        commission_per_share=0.005,
        min_commission=1.0,
        slippage_pct=0.001,
        spread_pct=0.0005,
        min_trade_frequency=0.5,
    )

    # Define strategy ideas to test
    strategies = [
        # SMA Cross variants
        SMACross(sma_short=5, sma_long=10),
        SMACross(sma_short=10, sma_long=20),
        SMACross(sma_short=20, sma_long=50),  # baseline
        SMACross(sma_short=50, sma_long=200),
        # RSI Mean Reversion
        RSIMeanReversion(rsi_period=14, oversold=30, overbought=70),
        RSIMeanReversion(rsi_period=14, oversold=20, overbought=80),
        RSIMeanReversion(rsi_period=7, oversold=25, overbought=75),
        # MACD Momentum
        MACDMomentum(fast=12, slow=26, signal=9),
        MACDMomentum(fast=8, slow=21, signal=5),
        # Bollinger Bands
        BollingerBands(window=20, num_std=2),
        BollingerBands(window=20, num_std=1.5),
        BollingerBands(window=10, num_std=2),
        # Z-Score Mean Reversion
        ZScoreMeanReversion(window=20, entry_z=2.0, exit_z=0.5),
        ZScoreMeanReversion(window=20, entry_z=1.5, exit_z=0.0),
        ZScoreMeanReversion(window=10, entry_z=2.0, exit_z=0.5),
        # Dual Momentum
        DualMomentum(sma_window=50),
        DualMomentum(sma_window=100),
        DualMomentum(sma_window=200),
    ]

    print("=" * 100)
    print("STRATEGY BACKTEST — Walk-Forward Evaluation")
    print("=" * 100)
    print(f"Symbols: {symbols}")
    print(f"Data source: Alpaca (primary), yfinance (fallback)")
    print(f"Cost model: $0.005/share, 0.1% slippage, 0.05% spread")
    print(f"Min trade frequency: 0.5 trades/day")
    print(f"Strategies: {len(strategies)}")
    print("=" * 100)
    print()

    # Fetch data for all symbols
    data = {}
    for symbol in symbols:
        df = pd.DataFrame()
        if dp.data_client:
            try:
                end = datetime.now()
                start = end - timedelta(days=730)
                df = dp.fetch_alpaca_bars(symbol, start, end)
            except Exception:
                pass
        if df.empty:
            df = dp.fetch_yf_history(symbol, period="2y", interval="1d")
        data[symbol] = df
        print(f"  {symbol}: {len(df)} bars from {'alpaca' if not df.empty and len(df) > 400 else 'yfinance'}")

    print()
    print("-" * 100)

    # Evaluate each strategy
    all_results = {}
    for strategy in strategies:
        print()
        print(strategy.name)
        print("-" * 60)
        strategy_results = []
        for symbol in symbols:
            df = data[symbol]
            if df.empty:
                continue
            # Split into periods
            train_end = min(252, int(len(df) * 0.5))
            val_end = min(378, int(len(df) * 0.75))
            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            oos_df = df.iloc[val_end:]
            for period_name, period_df in [("train", train_df), ("val", val_df), ("oos", oos_df)]:
                if len(period_df) < 60:
                    continue
                result = evaluator.evaluate(
                    candidate_id=strategy.name,
                    strategy=strategy,
                    df=period_df,
                    symbol=symbol,
                    period=period_name,
                )
                strategy_results.append(result)

        # Aggregate results
        if strategy_results:
            avg_return = np.mean([r.net_return_pct for r in strategy_results])
            avg_sharpe = np.mean([r.sharpe_ratio for r in strategy_results])
            avg_dd = np.mean([r.max_drawdown_pct for r in strategy_results])
            avg_trades = np.mean([r.total_trades for r in strategy_results])
            avg_freq = np.mean([r.trade_frequency for r in strategy_results])
            passed = sum(1 for r in strategy_results if r.passed_gates)
            total = len(strategy_results)
            avg_sma_sharpe = np.mean([r.baseline_sma_sharpe for r in strategy_results])
            avg_bh_sharpe = np.mean([r.baseline_buyhold_sharpe for r in strategy_results])

            status = "PASS" if passed > 0 else "FAIL"
            print(f"  Avg Return:     {avg_return:>8.2f}%")
            print(f"  Avg Sharpe:     {avg_sharpe:>8.2f}  (SMA: {avg_sma_sharpe:.2f}, B&H: {avg_bh_sharpe:.2f})")
            print(f"  Avg Max DD:     {avg_dd:>8.2f}%")
            print(f"  Avg Trades:     {avg_trades:>8.1f}")
            print(f"  Avg Freq:       {avg_freq:>8.3f} trades/day")
            print(f"  Gates Passed:   {passed}/{total}  [{status}]")

            if passed > 0:
                print(f"  >>> PASSED GATES <<<")
                for r in strategy_results:
                    if r.passed_gates:
                        print(f"      {r.symbol} [{r.period}]: return={r.net_return_pct:.2f}%, sharpe={r.sharpe_ratio:.2f}, trades={r.total_trades}")

            all_results[strategy.name] = {
                "results": strategy_results,
                "avg_return": avg_return,
                "avg_sharpe": avg_sharpe,
                "avg_dd": avg_dd,
                "avg_trades": avg_trades,
                "avg_freq": avg_freq,
                "passed": passed,
                "total": total,
            }

    # Summary
    print()
    print("=" * 100)
    print("SUMMARY — Strategies Ranked by Avg Sharpe")
    print("=" * 100)
    ranked = sorted(all_results.items(), key=lambda x: x[1]["avg_sharpe"], reverse=True)
    print(f"{'Rank':<5} {'Strategy':<45} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'Freq':>8} {'Pass':>6}")
    print("-" * 100)
    for i, (name, data) in enumerate(ranked, 1):
        status = "✓" if data["passed"] > 0 else "✗"
        print(f"{i:<5} {name:<45} {data['avg_return']:>7.2f}% {data['avg_sharpe']:>7.2f} {data['avg_dd']:>7.2f}% {data['avg_trades']:>7.1f} {data['avg_freq']:>7.3f} {status:>6}")

    print()
    print("=" * 100)
    print("BASELINES")
    print("=" * 100)
    # Compute overall baselines
    all_sma_sharpe = [r.baseline_sma_sharpe for results in all_results.values() for r in results["results"]]
    all_bh_sharpe = [r.baseline_buyhold_sharpe for results in all_results.values() for r in results["results"]]
    all_sma_return = [r.baseline_sma_return_pct for results in all_results.values() for r in results["results"]]
    all_bh_return = [r.baseline_buyhold_return_pct for results in all_results.values() for r in results["results"]]
    print(f"  SMA Cross 20/50:  Avg Return {np.mean(all_sma_return):.2f}%, Avg Sharpe {np.mean(all_sma_sharpe):.2f}")
    print(f"  Buy & Hold:       Avg Return {np.mean(all_bh_return):.2f}%, Avg Sharpe {np.mean(all_bh_sharpe):.2f}")
    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print("  ✓ = passed at least one gate (data quality, trade frequency, baseline beat, profit factor)")
    print("  ✗ = failed all gates")
    print("  Strategies that pass gates are candidates for paper trading evaluation.")
    print("  No strategy is promoted automatically — operator approval required.")
    print("=" * 100)


if __name__ == "__main__":
    main()
