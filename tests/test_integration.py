"""Integration tests: end-to-end verification of the combined bot.

These tests run the actual CLI commands and verify they produce
valid output. They exercise the real data layer (yfinance + Alpaca).
"""
import os
import sys
import unittest
from io import StringIO

# Ensure finance-bot is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import DataProvider
from strategies.baseline import SMACrossStrategy
from risk import RiskManager
from journal.trade_log import log_trade, ensure_header, LOG_PATH
from tests._external import yf_or_skip


SAMPLE_CONFIG = {
    "risk_limits": {
        "per_trade_max_loss_pct": 2.0,
        "per_trade_max_loss_usd": 2000.0,
        "total_open_exposure_pct": 10.0,
        "total_open_exposure_usd": 10000.0,
        "max_position_size_pct": 5.0,
        "max_position_size_usd": 5000.0,
        "daily_loss_limit_pct": 5.0,
        "daily_loss_limit_usd": 5000.0,
        "drawdown_halt_pct": 15.0,
        "drawdown_halt_usd": 15000.0,
        "min_order_collateral_usd": 15.0,
        "max_leverage": 1.0,
    },
    "market_universe": ["SPY", "QQQ"],
    "strategy": {
        "sma_short": 20,
        "sma_long": 50,
    },
}


class TestEndToEndDataFlow(unittest.TestCase):
    """End-to-end: fetch data → generate signal → check risk → log trade."""

    def setUp(self):
        self.dp = DataProvider(paper=True)
        self.strategy = SMACrossStrategy(sma_short=20, sma_long=50)
        self.rm = RiskManager(SAMPLE_CONFIG, initial_balance=100000)

    def test_yfinance_data_feeds_strategy(self):
        """yfinance data should be sufficient for SMA Cross backtest."""
        df = yf_or_skip(self, lambda: self.dp.fetch_yf_history("SPY", period="1y", interval="1d"), "SPY 1y")
        self.assertIn("Close", df.columns)
        result = self.strategy.backtest(df)
        self.assertIn("total_return_pct", result)
        self.assertIn("total_trades", result)

    def test_signal_generation_from_real_data(self):
        """Signal should be generated from real market data."""
        df = yf_or_skip(self, lambda: self.dp.fetch_yf_history("QQQ", period="60d", interval="1d"), "QQQ 60d")
        signal = self.strategy.generate_signal(df)
        self.assertIn(signal, [-1, 0, 1])

    def test_risk_check_allows_normal_trade(self):
        """Risk manager should allow a normal-sized trade (fixed price: this tests the
        risk rules, not the data feed, so it must not depend on Yahoo being up)."""
        close = 600.0
        stop = close * 0.97
        # Small qty that fits within position size limit (5% of $100k = $5k)
        qty = min(round(1000 / close, 0), 10)  # keep notional under $1k
        check = self.rm.check_trade(
            symbol="SPY", side="buy", qty=qty,
            entry_price=close, stop_price=stop,
        )
        self.assertTrue(check.passed, f"Trade should pass: {check.reason}")


class TestJournalIntegration(unittest.TestCase):
    """Test that journal logging works end-to-end."""

    def test_log_trade_creates_entry(self):
        """log_trade should append a valid row to the CSV."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                         prefix="test_journal_") as tmp:
            tmp_path = tmp.name
        # Remove the empty file so ensure_header creates it fresh with header
        os.unlink(tmp_path)
        try:
            row = log_trade(
                symbol="TEST", action="buy", qty=10, price=100.0,
                reason="test_entry", source="test", status="simulated",
                path=tmp_path,
            )
            self.assertEqual(len(row), 9)
            self.assertEqual(row[1], "TEST")
            self.assertEqual(row[2], "buy")
            # File should have header + 1 data row = 2 lines
            with open(tmp_path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_journal_schema(self):
        """Journal should use the documented schema."""
        from journal.trade_log import HEADERS
        expected = ["timestamp", "symbol", "action", "qty", "price",
                     "order_id", "reason", "source", "status"]
        self.assertEqual(HEADERS, expected)


class TestCombinedConfig(unittest.TestCase):
    """Verify the combined config.yaml has all required sections."""

    def test_config_loads(self):
        import yaml
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.yaml",
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.assertIn("market_universe", cfg)
        self.assertIn("alpaca", cfg)
        self.assertIn("yfinance", cfg)
        self.assertIn("strategy", cfg)
        self.assertIn("risk_limits", cfg)
        self.assertIn("schedule", cfg)
        self.assertIn("journal", cfg)
        # Liquid config should exist (disabled)
        self.assertIn("liquid", cfg)

    def test_config_risk_limits(self):
        import yaml
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.yaml",
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        limits = cfg["risk_limits"]
        self.assertIn("per_trade_max_loss_pct", limits)
        self.assertIn("total_open_exposure_pct", limits)
        self.assertIn("daily_loss_limit_pct", limits)
        self.assertIn("drawdown_halt_pct", limits)
        self.assertIn("max_leverage", limits)


class TestAlpacaConnection(unittest.TestCase):
    """Verify Alpaca paper trading connection works."""

    @unittest.skipUnless(
        os.getenv("APCA_API_KEY_ID") and os.getenv("APCA_API_SECRET_KEY"),
        "Alpaca credentials not set",
    )
    def test_account_active(self):
        """Alpaca paper account should be ACTIVE."""
        dp = DataProvider(paper=True)
        account = dp.get_account()
        self.assertEqual(account["status"].value if hasattr(account["status"], "value")
                         else account["status"], "ACTIVE")

    @unittest.skipUnless(
        os.getenv("APCA_API_KEY_ID") and os.getenv("APCA_API_SECRET_KEY"),
        "Alpaca credentials not set",
    )
    def test_paper_mode(self):
        """Trading client should be in paper mode."""
        dp = DataProvider(paper=True)
        self.assertTrue(dp.paper)
