"""Tests for the RiskManager."""
import unittest
from datetime import date

from risk import RiskManager, RiskCheck


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
}


class TestRiskManagerInit(unittest.TestCase):
    def test_init_defaults(self):
        rm = RiskManager(SAMPLE_CONFIG, initial_balance=100000)
        self.assertEqual(rm.balance, 100000)
        self.assertEqual(rm.peak_balance, 100000)
        self.assertFalse(rm.halted)

    def test_today_property(self):
        rm = RiskManager(SAMPLE_CONFIG)
        self.assertEqual(rm.today, date.today().isoformat())


class TestRiskChecks(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(SAMPLE_CONFIG, initial_balance=100000)

    def test_trade_passes_all_limits(self):
        """A small trade should pass all risk checks."""
        check = self.rm.check_trade(
            symbol="AAPL", side="buy", qty=10,
            entry_price=200, stop_price=195,
        )
        self.assertTrue(check.passed)
        self.assertEqual(check.action, "allow")

    def test_reject_small_order(self):
        """Order below min collateral should be rejected."""
        check = self.rm.check_trade(
            symbol="AAPL", side="buy", qty=1,
            entry_price=10, stop_price=9,
        )
        self.assertFalse(check.passed)
        self.assertEqual(check.action, "reject")
        self.assertIn("min", check.reason.lower())

    def test_reject_oversized_risk(self):
        """Per-trade risk exceeding limit should be rejected."""
        check = self.rm.check_trade(
            symbol="AAPL", side="buy", qty=1000,
            entry_price=200, stop_price=195,  # risk = $5000 > $2000 limit
        )
        self.assertFalse(check.passed)
        self.assertIn("per-trade", check.reason.lower())

    def test_reject_position_too_large(self):
        """Position exceeding max size should be rejected."""
        # Use high entry and tight stop so risk-per-trade is within limit
        # but notional is large enough to exceed position size cap.
        check = self.rm.check_trade(
            symbol="AAPL", side="buy", qty=100,
            entry_price=200, stop_price=199.50,  # risk = $50, within $2000 limit
        )  # notional = $20,000 > $5,000 max position size
        self.assertFalse(check.passed)
        self.assertIn("position size", check.reason.lower())

    def test_reject_exceeds_total_exposure(self):
        """Trade pushing total exposure over limit should be rejected."""
        # First fill up near the exposure limit with a low-risk trade
        self.rm.record_position("SPY", 9800)
        # Now add a small-risk trade that pushes total exposure over $10,000
        check = self.rm.check_trade(
            symbol="AAPL", side="buy", qty=20,
            entry_price=200, stop_price=199,  # risk = $20, within $2000 limit
        )  # notional = $4,000, total = $13,800 > $10,000 limit
        self.assertFalse(check.passed)
        self.assertIn("exposure", check.reason.lower())


class TestDailyLossLimit(unittest.TestCase):
    def test_daily_loss_triggers_halt(self):
        rm = RiskManager(SAMPLE_CONFIG, initial_balance=100000)
        # Simulate hitting daily loss limit
        rm._daily[rm.today].realized_pl = -6000
        check = rm.check_trade(
            symbol="AAPL", side="buy", qty=10,
            entry_price=200, stop_price=199,
        )
        self.assertFalse(check.passed)
        self.assertEqual(check.action, "halt")

    def test_daily_loss_within_limit(self):
        rm = RiskManager(SAMPLE_CONFIG, initial_balance=100000)
        # Small loss, within limit
        rm._daily[rm.today].realized_pl = -1000
        check = rm.check_trade(
            symbol="AAPL", side="buy", qty=10,
            entry_price=200, stop_price=195,
        )
        self.assertTrue(check.passed)


class TestDrawdownHalt(unittest.TestCase):
    def test_drawdown_triggers_halt(self):
        rm = RiskManager(SAMPLE_CONFIG, initial_balance=100000)
        # Drop balance below 15% drawdown ($85k threshold)
        rm.update_balance(80000)
        check = rm.check_trade(
            symbol="AAPL", side="buy", qty=10,
            entry_price=200, stop_price=195,
        )
        self.assertFalse(check.passed)
        self.assertTrue(rm.halted)
        self.assertEqual(check.action, "halt")


class TestLeverageCheck(unittest.TestCase):
    def test_leverage_within_limit(self):
        rm = RiskManager(SAMPLE_CONFIG)
        check = rm.check_max_leverage(1.0)
        self.assertTrue(check.passed)

    def test_leverage_exceeds_limit(self):
        rm = RiskManager(SAMPLE_CONFIG)
        check = rm.check_max_leverage(3.0)
        self.assertFalse(check.passed)


class TestDailySummary(unittest.TestCase):
    def test_summary_structure(self):
        rm = RiskManager(SAMPLE_CONFIG, initial_balance=100000)
        summary = rm.daily_summary()
        expected_keys = {
            "date", "realized_pl", "trade_count", "loss",
            "daily_loss_limit_usd", "halted", "current_balance",
            "peak_balance", "total_exposure",
        }
        self.assertTrue(expected_keys.issubset(summary.keys()))
        self.assertEqual(summary["trade_count"], 0)
        self.assertFalse(summary["halted"])


class TestHaltedTrading(unittest.TestCase):
    def test_halted_blocks_new_trades(self):
        rm = RiskManager(SAMPLE_CONFIG, initial_balance=100000)
        rm.halted = True
        rm.halt_reason = "Test halt"
        check = rm.check_trade(
            symbol="AAPL", side="buy", qty=10,
            entry_price=200, stop_price=195,
        )
        self.assertFalse(check.passed)
        self.assertEqual(check.action, "halt")
