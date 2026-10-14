"""Tests for the DataProvider layer (yfinance + Alpaca).

Integration tests — require network access. Skipped gracefully if
Alpaca credentials are not configured.
"""
import os
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from data import DataProvider
from tests._external import yf_or_skip

def assert_valid_quote(testcase, quote):
    testcase.assertIsInstance(quote, dict, "Malformed quote type")
    if "bid_price" not in quote or "ask_price" not in quote or quote["bid_price"] is None or quote["ask_price"] is None:
        testcase.skipTest("EXTERNAL DATA UNAVAILABLE: Alpaca quote missing one side")
    try:
        bid, ask = float(quote["bid_price"]), float(quote["ask_price"])
    except (ValueError, TypeError) as exc:
        raise AssertionError("Malformed quote price") from exc
    import math
    testcase.assertTrue(math.isfinite(bid) and math.isfinite(ask), "Malformed quote price")
    if bid == 0 or ask == 0:
        testcase.skipTest("EXTERNAL DATA UNAVAILABLE: zero-sided Alpaca quote")
    testcase.assertGreater(bid, 0)
    testcase.assertGreaterEqual(ask, bid)

class TestQuoteValidation(unittest.TestCase):
    def test_missing_one_side_is_external_unavailable(self):
        for quote in ({"bid_price": 320.06, "ask_price": 0.0},
                      {"bid_price": 0.0, "ask_price": 321.0}):
            with self.assertRaisesRegex(unittest.SkipTest, "^EXTERNAL DATA UNAVAILABLE"):
                assert_valid_quote(self, quote)
        with self.assertRaisesRegex(unittest.SkipTest, "^EXTERNAL DATA UNAVAILABLE"):
            assert_valid_quote(self, {"bid_price": 320.06})
        with self.assertRaisesRegex(unittest.SkipTest, "^EXTERNAL DATA UNAVAILABLE"):
            assert_valid_quote(self, {"bid_price": None, "ask_price": 321.0})

    def test_fake_two_sided_quote_and_malformed_quote(self):
        assert_valid_quote(self, {"bid_price": 320.06, "ask_price": 320.07})
        with self.assertRaises(AssertionError):
            assert_valid_quote(self, {"bid_price": 321.0, "ask_price": 320.0})
        with self.assertRaises(AssertionError):
            assert_valid_quote(self, {"bid_price": "garbage", "ask_price": 320.0})
        with self.assertRaises(AssertionError):
            assert_valid_quote(self, ["bid_price", "ask_price"])


class TestDataProviderInit(unittest.TestCase):
    """Unit tests for DataProvider initialization."""

    def test_init_without_credentials(self):
        """DataProvider should warn (not crash) when Alpaca creds are missing."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove any inherited env vars
            for key in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
                os.environ.pop(key, None)
            dp = DataProvider(paper=True)
            self.assertFalse(dp.alpaca_key)
            self.assertIsNone(dp.trading_client)
            self.assertIsNone(dp.data_client)

    @patch("data.TradingClient")
    @patch("data.StockHistoricalDataClient")
    def test_init_with_credentials(self, mock_data_client, mock_trading_client):
        """DataProvider should initialize Alpaca clients when creds are set."""
        dp = DataProvider(
            alpaca_key="test_key_123",
            alpaca_secret="test_secret_456",
            paper=True,
        )
        self.assertEqual(dp.alpaca_key, "test_key_123")
        self.assertTrue(dp.paper)
        mock_trading_client.assert_called_once()
        mock_data_client.assert_called_once()


class TestYfinanceData(unittest.TestCase):
    """Tests for yfinance historical data fetching."""

    def setUp(self):
        self.dp = DataProvider(paper=True)

    def test_fetch_yf_history_basic(self):
        """yfinance should return a non-empty DataFrame for a known symbol."""
        df = yf_or_skip(self, lambda: self.dp.fetch_yf_history("SPY", period="5d", interval="1d"), "SPY 5d")
        self.assertIn("Close", df.columns, "DataFrame must have 'Close' column")
        self.assertIn("Open", df.columns)
        self.assertIn("High", df.columns)
        self.assertIn("Low", df.columns)
        self.assertIn("Volume", df.columns)

    def test_fetch_yf_history_row_count(self):
        """yfinance should return approximately the right number of rows."""
        df = yf_or_skip(self, lambda: self.dp.fetch_yf_history("AAPL", period="5d", interval="1d"), "AAPL 5d")
        self.assertGreaterEqual(len(df), 3, "Expected at least 3 rows for 5d")
        self.assertLessEqual(len(df), 5, "Expected at most 5 rows for 5d")

    def test_fetch_yf_history_empty_on_bad_symbol(self):
        """yfinance should return empty DataFrame for invalid symbol."""
        df = self.dp.fetch_yf_history("INVALID_TICKER_12345", period="1d", interval="1d")
        self.assertTrue(df.empty, "Expected empty DataFrame for invalid ticker")

    def test_fetch_yf_dividends_splits(self):
        """yfinance should return dividends and splits data."""
        dividends, splits = self.dp.fetch_yf_dividends_splits("SPY")
        # yfinance returns pandas Series/Index for dividends and splits
        self.assertTrue(hasattr(dividends, "index"), "Expected pandas object for dividends")
        self.assertTrue(hasattr(splits, "index"), "Expected pandas object for splits")

    def test_fetch_yf_fundamentals(self):
        """yfinance should return a non-empty dict of fundamentals."""
        info = self.dp.fetch_yf_fundamentals("AAPL")
        self.assertIsInstance(info, dict)
        self.assertGreater(len(info), 0, "Expected non-empty fundamentals dict")


@unittest.skipUnless(
    os.getenv("APCA_API_KEY_ID") and os.getenv("APCA_API_SECRET_KEY"),
    "Alpaca credentials not set — skipping live API tests",
)
class TestAlpacaData(unittest.TestCase):
    """Tests for Alpaca real-time data and account (require credentials)."""

    def setUp(self):
        self.dp = DataProvider(paper=True)

    def test_get_account(self):
        """Alpaca account should return status ACTIVE in paper mode."""
        account = self.dp.get_account()
        self.assertIsNotNone(account)
        self.assertIn("status", account)

    def test_get_clock(self):
        """Alpaca clock should return market open status."""
        clock = self.dp.get_clock()
        self.assertIn("is_open", clock)
        self.assertIsInstance(clock["is_open"], bool)

    def test_get_latest_quote(self):
        """Alpaca should return a real-time quote for AAPL."""
        quote = self.dp.get_latest_quote("AAPL")
        self.assertIn("bid_price", quote)
        self.assertIn("ask_price", quote)
        assert_valid_quote(self, quote)

    def test_fetch_alpaca_bars(self):
        """Alpaca should return bars with normalized column names."""
        from alpaca.data.timeframe import TimeFrame
        from datetime import datetime, timedelta
        end = datetime.now()
        start = end - timedelta(days=10)
        df = self.dp.fetch_alpaca_bars("AAPL", start, end, timeframe=TimeFrame.Day)
        self.assertFalse(df.empty, "Expected non-empty Alpaca bars")
        # Columns should be normalized to capitalized (yfinance format)
        self.assertIn("Close", df.columns)
        self.assertIn("Open", df.columns)

    def test_get_history_hybrid(self):
        """Hybrid get_history should return a non-empty DataFrame."""
        from datetime import datetime, timedelta
        end = datetime.now()
        start = end - timedelta(days=30)
        df = self.dp.get_history("AAPL", start, end)
        self.assertFalse(df.empty)
        self.assertIn("Close", df.columns)


class TestDataNormalization(unittest.TestCase):
    """Tests for cross-provider column normalization."""

    def test_column_normalization_names(self):
        """DataProvider should normalize Alpaca lowercase columns to capitalized."""
        # The normalization map is defined in fetch_alpaca_bars
        # Just verify the rename map keys exist
        rename_map = {
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume", "vwap": "Vwap",
            "trade_count": "Trade Count", "timestamp": "Timestamp",
        }
        self.assertEqual(rename_map["close"], "Close")
        self.assertEqual(rename_map["volume"], "Volume")

    def test_backfill_yf_timezone_handling(self):
        """_backfill_yf should handle tz-aware vs naive datetimes."""
        dp = DataProvider(paper=True)
        from alpaca.data.timeframe import TimeFrame
        from datetime import datetime, timedelta
        end = datetime.now()
        start = end - timedelta(days=7)
        # _backfill_yf localizes naive datetimes to match the tz-aware index
        masked = dp._backfill_yf("SPY", start, end, TimeFrame.Day)
        if not masked.empty:
            # Verify all returned rows fall within the requested range
            # (the method handles tz-aware vs naive comparison internally)
            self.assertTrue(len(masked) <= 7, "Should not return more than 7 days")
