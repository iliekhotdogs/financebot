"""
Research roles — specialist analysis feeding Jev.

Each role analyzes the point-in-time snapshot and returns concise,
structured findings with evidence, timestamps, and data sources.
Roles advise Jev; they do NOT place orders or override decisions.

Roles:
  - TechnicalRole: SMA, RSI, MACD, ATR, support/resistance
  - FundamentalRole: P/E, market cap, sector, earnings
  - NewsSentimentRole: news headlines and sentiment (if configured)
  - BullBearRole: bullish/bearish thesis with evidence
"""

from .roles import (
    ResearchFinding,
    TechnicalRole,
    FundamentalRole,
    NewsSentimentRole,
    BullBearRole,
    ResearchAggregator,
)

__all__ = [
    "ResearchFinding",
    "TechnicalRole",
    "FundamentalRole",
    "NewsSentimentRole",
    "BullBearRole",
    "ResearchAggregator",
]
