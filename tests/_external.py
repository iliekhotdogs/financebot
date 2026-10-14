"""Helpers for the few tests that call the live yfinance service.

Yahoo sometimes returns an empty frame (rate limit, outage: "possibly
delisted; no price data found"). That says nothing about the bot's code, so
these tests retry, then SKIP with an explicit reason instead of failing the
install gate. Wrong or malformed data still fails them. The bot's own guard
for missing data (engine_etf: incomplete price data -> no trading) is tested
offline in tests/test_etf_trend.py.
"""
import time

TRIES = 3


def yf_or_skip(testcase, fetch, what: str):
    """Return fetch() once it is non-empty; skip the test after TRIES empty results."""
    for attempt in range(TRIES):
        df = fetch()
        if df is not None and len(df) > 0:
            return df
        if attempt + 1 < TRIES:
            time.sleep(2 * (attempt + 1))
    testcase.skipTest(f"EXTERNAL DATA UNAVAILABLE: yfinance returned no rows for {what} after {TRIES} tries "
                      f"(Yahoo outage or rate limit, not a bot-code failure)")
