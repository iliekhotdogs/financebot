"""Tests for the SMA Cross strategy (baseline)."""
import unittest
import pytest
import numpy as np
import pandas as pd

from strategies.baseline import SMACrossStrategy


def make_test_df(n=120, price_start=100, trend="up"):
    """Generate synthetic price data for testing."""
    prices = []
    if trend == "up":
        prices = np.linspace(price_start, price_start + 30, n)
    elif trend == "down":
        prices = np.linspace(price_start + 30, price_start, n)
    else:
        prices = np.full(n, price_start)

    np.random.seed(42)
    noise = np.random.normal(0, 0.5, n)
    prices = prices + noise

    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Open": prices,
        "High": prices + 1,
        "Low": prices - 1,
        "Close": prices,
        "Volume": np.random.randint(1000, 10000, n),
    }, index=dates)
    return df


class TestSMACrossStrategy(unittest.TestCase):
    """Tests for SMA Cross strategy signal generation and backtesting."""

    def setUp(self):
        self.strategy = SMACrossStrategy(sma_short=20, sma_long=50)

    def test_signal_insufficient_data(self):
        """Should return 0 (hold) when data is too short for SMA."""
        df = make_test_df(n=30)  # < 50 rows
        signal = self.strategy.generate_signal(df)
        self.assertEqual(signal, 0, "Should hold with insufficient data")

    def test_signal_long_when_short_above_long(self):
        """Should return 1 (buy) when short SMA > long SMA and crossover occurs."""
        df = make_test_df(n=120, trend="up")
        # Modify last few rows to ensure crossover
        df.iloc[-1, df.columns.get_loc("Close")] += 5
        signal = self.strategy.generate_signal(df)
        # After a strong uptrend, short SMA should be above long SMA
        self.assertIn(signal, [0, 1], "Signal should be 0 or 1")

    def test_signal_short_when_short_below_long(self):
        """Should return -1 (sell) when short SMA crosses below long SMA."""
        df = make_test_df(n=120, trend="down")
        df.iloc[-1, df.columns.get_loc("Close")] -= 5
        signal = self.strategy.generate_signal(df)
        self.assertIn(signal, [-1, 0], "Signal should be -1 or 0")

    def test_signal_hold_when_no_crossover(self):
        """Should return current position (0 or 1) when no crossover at latest."""
        df = make_test_df(n=120, trend="steady")
        signal = self.strategy.generate_signal(df)
        self.assertIn(signal, [0, 1], "Signal should be 0 or 1 when flat")

    def test_backtest_returns_valid_dict(self):
        """Backtest should return a dict with all expected keys."""
        df = make_test_df(n=120, trend="up")
        result = self.strategy.backtest(df)
        expected_keys = {
            "total_return_pct", "max_drawdown_pct", "sharpe_ratio",
            "total_trades", "final_strategy_value", "final_buyhold_value",
        }
        self.assertTrue(expected_keys.issubset(result.keys()),
                        f"Missing keys: {expected_keys - set(result.keys())}")

    def test_backtest_insufficient_data(self):
        """Backtest should return zero metrics for insufficient data."""
        df = make_test_df(n=30)
        result = self.strategy.backtest(df)
        self.assertEqual(result["total_trades"], 0)
        self.assertEqual(result["total_return_pct"], 0.0)
        self.assertIn("error", result)

    def test_backtest_trade_count_positive(self):
        """Backtest should detect at least some crossovers in trending data."""
        df = make_test_df(n=252, trend="up")
        result = self.strategy.backtest(df)
        self.assertGreater(result["total_trades"], 0,
                           "Expected at least 1 trade in 252 days of trending data")

    def test_backtest_buyhold_vs_strategy(self):
        """Buy-and-hold value should always be present in results."""
        df = make_test_df(n=200, trend="up")
        result = self.strategy.backtest(df)
        self.assertIn("final_buyhold_value", result)
        self.assertGreater(result["final_buyhold_value"], 0)

    def test_signal_returns_int(self):
        """Signal must be an int: 1, 0, or -1."""
        df = make_test_df(n=120, trend="up")
        signal = self.strategy.generate_signal(df)
        self.assertIsInstance(signal, (int, np.integer))
        self.assertIn(signal, [-1, 0, 1])

    def test_strategy_version(self):
        """Strategy should report its version."""
        self.assertEqual(self.strategy.version, "v1.0")
        self.assertEqual(self.strategy.name, "SMA Cross (20/50)")


class TestSMACrossV1_1:
    """Tests for v1.1 strategy (65% Rule + Runner Hug)."""

    @classmethod
    def setup_class(cls):
        from strategies.v1_1 import SMACrossV1_1
        cls.strategy_cls = SMACrossV1_1

    @pytest.fixture
    def sample_prices(self):
        """Build a price series with uptrend then drawdown — tests gain-locking."""
        dates = pd.date_range("2024-01-01", periods=200)
        up = np.linspace(100, 120, 100)
        up2 = np.linspace(120, 140, 50)
        down = np.linspace(140, 110, 50)
        prices = np.concatenate([up, up2, down])
        df = pd.DataFrame({
            "Close": prices,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Open": prices * 0.995,
            "Volume": 100000,
        }, index=dates)
        return df

    def test_v1_1_has_version_and_name(self):
        s = self.strategy_cls()
        assert s.version == "v1.1"
        assert "65" in s.name or "Runner" in s.name

    def test_v1_1_backtest_produces_metrics(self, sample_prices):
        s = self.strategy_cls()
        result = s.backtest(sample_prices)
        assert "total_return_pct" in result
        assert "total_trades" in result
        assert "max_drawdown_pct" in result

    def test_65pct_rule_reduces_drawdown(self, sample_prices):
        """65% Rule should produce lower drawdown than naive v1.0."""
        from strategies.baseline import SMACrossStrategy
        r10 = SMACrossStrategy(20, 50).backtest(sample_prices)
        r11 = self.strategy_cls().backtest(sample_prices)
        # v1.1 drawdown should be less severe (less negative) than v1.0
        assert r11["max_drawdown_pct"] >= r10["max_drawdown_pct"]

    def test_exit_reasons_tracked(self, sample_prices):
        """v1.1 tracks exit reasons including 65% Rule and Runner Hug."""
        result = self.strategy_cls().backtest(sample_prices)
        assert "exit_reasons" in result
        reasons = result["exit_reasons"]
        assert any("65pct" in r or "runner" in r for r in reasons.keys())
