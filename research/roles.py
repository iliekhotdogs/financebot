"""
Research roles — specialist analysis for the advisory pipeline.

Each role receives the point-in-time snapshot and returns a concise,
structured ResearchFinding with evidence, timestamps, and data sources.
Roles advise Jev; they do NOT place orders or override decisions.

All roles are deterministic (no LLM calls) so they are fast, auditable,
and cannot fail due to API issues. They provide structured input that
Jev consumes alongside the advisor's recommendation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "journal", "cache")


def _cache_get(kind: str, key: str, ttl_seconds: int):
    """Small JSON disk cache so per-symbol lookups survive across cron cycles."""
    path = os.path.join(CACHE_DIR, kind, f"{key.replace('/', '_')}.json")
    try:
        if time.time() - os.path.getmtime(path) > ttl_seconds:
            return None
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _cache_put(kind: str, key: str, value) -> None:
    path = os.path.join(CACHE_DIR, kind, f"{key.replace('/', '_')}.json")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(value, f, default=str)
    except OSError:
        pass


def _to_float(v) -> Optional[float]:
    """yfinance sometimes returns numbers as strings ('Infinity', '12.3')."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


@dataclass
class ResearchFinding:
    """A single research finding from a specialist role."""
    role: str
    finding: str
    direction: str  # "bullish" | "bearish" | "neutral"
    confidence: float  # 0.0 — 1.0
    evidence: str
    data_sources: list = field(default_factory=list)
    timestamp: str = ""
    related_symbol: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "finding": self.finding,
            "direction": self.direction,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "data_sources": self.data_sources,
            "timestamp": self.timestamp,
            "related_symbol": self.related_symbol,
        }


class TechnicalRole:
    """
    Technical analysis role.

    Computes SMA, RSI, MACD, ATR, and support/resistance levels
    from the point-in-time snapshot data. Returns structured findings.
    """

    name = "technical"

    def __init__(self, rsi_period: int = 14, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def analyze(self, snapshot: dict, df: pd.DataFrame) -> list[ResearchFinding]:
        """Analyze snapshot and return technical findings."""
        findings = []
        symbol = snapshot.get("symbol", "")

        if df.empty or len(df) < 20:
            return [ResearchFinding(
                role=self.name,
                finding="Insufficient data for technical analysis",
                direction="neutral",
                confidence=0.0,
                evidence=f"Only {len(df)} bars available",
                data_sources=["snapshot"],
                related_symbol=symbol,
            )]

        # --- SMA analysis ---
        sma_short = snapshot.get("market", {}).get("sma_short")
        sma_long = snapshot.get("market", {}).get("sma_long")
        close = snapshot.get("market", {}).get("close", 0)

        if sma_short and sma_long:
            if sma_short > sma_long:
                direction = "bullish"
                finding = f"SMA bullish alignment: {sma_short:.2f} > {sma_long:.2f}"
                evidence = f"Short-term SMA ({sma_short:.2f}) above long-term SMA ({sma_long:.2f}) at close {close:.2f}"
                confidence = min(0.9, 0.5 + abs(sma_short - sma_long) / sma_long * 10)
            else:
                direction = "bearish"
                finding = f"SMA bearish alignment: {sma_short:.2f} < {sma_long:.2f}"
                evidence = f"Short-term SMA ({sma_short:.2f}) below long-term SMA ({sma_long:.2f}) at close {close:.2f}"
                confidence = min(0.9, 0.5 + abs(sma_short - sma_long) / sma_long * 10)

            findings.append(ResearchFinding(
                role=self.name,
                finding=finding,
                direction=direction,
                confidence=round(confidence, 2),
                evidence=evidence,
                data_sources=["alpaca_bars" if snapshot.get("market", {}).get("bars_count", 0) > 0 else "yfinance"],
                related_symbol=symbol,
            ))

        # --- RSI ---
        rsi = self._compute_rsi(df)
        if rsi is not None:
            if rsi > 70:
                direction = "bearish"
                finding = f"RSI overbought: {rsi:.1f}"
                evidence = f"RSI({self.rsi_period}) = {rsi:.1f} > 70, potential reversal zone"
                confidence = min(0.85, 0.5 + (rsi - 70) / 30)
            elif rsi < 30:
                direction = "bullish"
                finding = f"RSI oversold: {rsi:.1f}"
                evidence = f"RSI({self.rsi_period}) = {rsi:.1f} < 30, potential bounce zone"
                confidence = min(0.85, 0.5 + (30 - rsi) / 30)
            else:
                direction = "neutral"
                finding = f"RSI neutral: {rsi:.1f}"
                evidence = f"RSI({self.rsi_period}) = {rsi:.1f}, no extreme reading"
                confidence = 0.5

            findings.append(ResearchFinding(
                role=self.name,
                finding=finding,
                direction=direction,
                confidence=round(confidence, 2),
                evidence=evidence,
                data_sources=["price_data"],
                related_symbol=symbol,
            ))

        # --- MACD ---
        macd_line, signal_line, hist = self._compute_macd(df)
        if macd_line is not None:
            if macd_line > signal_line and hist > 0:
                direction = "bullish"
                finding = f"MACD bullish crossover"
                evidence = f"MACD ({macd_line:.3f}) > Signal ({signal_line:.3f}), histogram={hist:.3f}"
                confidence = min(0.8, 0.5 + abs(hist) / abs(macd_line) * 5 if macd_line != 0 else 0.5)
            elif macd_line < signal_line and hist < 0:
                direction = "bearish"
                finding = f"MACD bearish crossover"
                evidence = f"MACD ({macd_line:.3f}) < Signal ({signal_line:.3f}), histogram={hist:.3f}"
                confidence = min(0.8, 0.5 + abs(hist) / abs(macd_line) * 5 if macd_line != 0 else 0.5)
            else:
                direction = "neutral"
                finding = f"MACD neutral"
                evidence = f"MACD ({macd_line:.3f}) near Signal ({signal_line:.3f})"
                confidence = 0.4

            findings.append(ResearchFinding(
                role=self.name,
                finding=finding,
                direction=direction,
                confidence=round(confidence, 2),
                evidence=evidence,
                data_sources=["price_data"],
                related_symbol=symbol,
            ))

        # --- ATR (volatility) ---
        atr = self._compute_atr(df)
        if atr is not None:
            atr_pct = atr / close * 100 if close > 0 else 0
            if atr_pct > 3.0:
                direction = "bearish"
                finding = f"High volatility: ATR {atr_pct:.1f}%"
                evidence = f"ATR(14) = {atr:.2f} ({atr_pct:.1f}% of price), elevated volatility increases risk"
                confidence = min(0.7, 0.4 + atr_pct / 10)
            elif atr_pct < 1.0:
                direction = "neutral"
                finding = f"Low volatility: ATR {atr_pct:.1f}%"
                evidence = f"ATR(14) = {atr:.2f} ({atr_pct:.1f}% of price), low volatility may precede expansion"
                confidence = 0.5
            else:
                direction = "neutral"
                finding = f"Normal volatility: ATR {atr_pct:.1f}%"
                evidence = f"ATR(14) = {atr:.2f} ({atr_pct:.1f}% of price)"
                confidence = 0.5

            findings.append(ResearchFinding(
                role=self.name,
                finding=finding,
                direction=direction,
                confidence=round(confidence, 2),
                evidence=evidence,
                data_sources=["price_data"],
                related_symbol=symbol,
            ))

        # --- Support/Resistance ---
        support, resistance = self._compute_support_resistance(df)
        if support and resistance:
            if close < support * 1.02:
                direction = "bullish"
                finding = f"Near support: {support:.2f}"
                evidence = f"Close {close:.2f} near support {support:.2f} (within 2%)"
                confidence = 0.65
            elif close > resistance * 0.98:
                direction = "bearish"
                finding = f"Near resistance: {resistance:.2f}"
                evidence = f"Close {close:.2f} near resistance {resistance:.2f} (within 2%)"
                confidence = 0.65
            else:
                direction = "neutral"
                finding = f"Between support/resistance"
                evidence = f"Close {close:.2f} between support {support:.2f} and resistance {resistance:.2f}"
                confidence = 0.5

            findings.append(ResearchFinding(
                role=self.name,
                finding=finding,
                direction=direction,
                confidence=round(confidence, 2),
                evidence=evidence,
                data_sources=["price_data"],
                related_symbol=symbol,
            ))

        return findings

    def _compute_rsi(self, df: pd.DataFrame) -> Optional[float]:
        if len(df) < self.rsi_period + 1:
            return None
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return None if pd.isna(val) else float(val)

    def _compute_macd(self, df: pd.DataFrame) -> tuple:
        if len(df) < self.macd_slow + self.macd_signal:
            return None, None, None
        ema_fast = df["Close"].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        hist = macd_line - signal_line
        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(hist.iloc[-1])

    def _compute_atr(self, df: pd.DataFrame) -> Optional[float]:
        if len(df) < 14:
            return None
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        val = atr.iloc[-1]
        return None if pd.isna(val) else float(val)

    def _compute_support_resistance(self, df: pd.DataFrame, lookback: int = 20) -> tuple:
        if len(df) < lookback:
            return None, None
        recent = df.tail(lookback)
        support = float(recent["Low"].min())
        resistance = float(recent["High"].max())
        return support, resistance


class FundamentalRole:
    """
    Fundamental analysis role.

    Fetches and analyzes fundamental data from yfinance.
    Returns structured findings about valuation and financial health.
    """

    name = "fundamental"

    def __init__(self, data_provider=None, cache_ttl_seconds: int = 86400):
        self.data_provider = data_provider
        self.cache_ttl_seconds = cache_ttl_seconds

    def analyze(self, snapshot: dict, df: pd.DataFrame) -> list[ResearchFinding]:
        """Analyze snapshot and return fundamental findings."""
        findings = []
        symbol = snapshot.get("symbol", "")

        if not self.data_provider:
            return [ResearchFinding(
                role=self.name,
                finding="Fundamental data provider not configured",
                direction="neutral",
                confidence=0.0,
                evidence="No data provider available for fundamental analysis",
                data_sources=[],
                related_symbol=symbol,
            )]

        try:
            info = _cache_get("fundamentals", symbol, self.cache_ttl_seconds)
            if info is None:
                info = self.data_provider.fetch_yf_fundamentals(symbol) or {}
                # Fundamentals don't change intraday: one yfinance call per
                # symbol per day instead of one per 5-minute cycle.
                _cache_put("fundamentals", symbol, {
                    k: info.get(k) for k in
                    ("trailingPE", "forwardPE", "marketCap", "dividendYield", "sector")
                })
        except Exception as e:
            return [ResearchFinding(
                role=self.name,
                finding=f"Fundamental data fetch failed: {e}",
                direction="neutral",
                confidence=0.0,
                evidence=str(e),
                data_sources=["yfinance"],
                related_symbol=symbol,
            )]

        if not info:
            return [ResearchFinding(
                role=self.name,
                finding="No fundamental data available",
                direction="neutral",
                confidence=0.0,
                evidence="yfinance returned empty data",
                data_sources=["yfinance"],
                related_symbol=symbol,
            )]

        # P/E ratio
        pe = _to_float(info.get("trailingPE")) or _to_float(info.get("forwardPE"))
        if pe and pe > 0:
            if pe < 15:
                direction = "bullish"
                finding = f"Attractive valuation: P/E {pe:.1f}"
                evidence = f"Trailing P/E of {pe:.1f} is below market average (~20)"
                confidence = min(0.7, 0.4 + (20 - pe) / 20)
            elif pe > 30:
                direction = "bearish"
                finding = f"Rich valuation: P/E {pe:.1f}"
                evidence = f"Trailing P/E of {pe:.1f} is above market average, limited margin of safety"
                confidence = min(0.7, 0.4 + (pe - 20) / 30)
            else:
                direction = "neutral"
                finding = f"Fair valuation: P/E {pe:.1f}"
                evidence = f"Trailing P/E of {pe:.1f} is near market average"
                confidence = 0.5

            findings.append(ResearchFinding(
                role=self.name,
                finding=finding,
                direction=direction,
                confidence=round(confidence, 2),
                evidence=evidence,
                data_sources=["yfinance"],
                related_symbol=symbol,
            ))

        # Market cap
        mcap = _to_float(info.get("marketCap"))
        if mcap:
            if mcap > 200e9:
                finding = f"Large cap: ${mcap/1e9:.0f}B market cap"
                evidence = f"Market cap ${mcap/1e9:.0f}B indicates established large-cap stock"
                confidence = 0.6
            elif mcap > 10e9:
                finding = f"Mid cap: ${mcap/1e9:.0f}B market cap"
                evidence = f"Market cap ${mcap/1e9:.0f}B indicates mid-cap stock"
                confidence = 0.5
            else:
                finding = f"Small cap: ${mcap/1e9:.1f}B market cap"
                evidence = f"Market cap ${mcap/1e9:.1f}B indicates small-cap stock, higher risk"
                confidence = 0.5

            findings.append(ResearchFinding(
                role=self.name,
                finding=finding,
                direction="neutral",
                confidence=round(confidence, 2),
                evidence=evidence,
                data_sources=["yfinance"],
                related_symbol=symbol,
            ))

        # Dividend yield
        div_yield = _to_float(info.get("dividendYield"))
        # yfinance >= 0.2.5x reports dividendYield in percent (0.45 == 0.45%)
        if div_yield is not None and div_yield > 1:
            div_yield = div_yield / 100
        if div_yield:
            if div_yield > 0.03:
                direction = "bullish"
                finding = f"High dividend yield: {div_yield*100:.1f}%"
                evidence = f"Dividend yield of {div_yield*100:.1f}% provides income cushion"
                confidence = 0.6
            elif div_yield > 0:
                direction = "neutral"
                finding = f"Moderate dividend yield: {div_yield*100:.1f}%"
                evidence = f"Dividend yield of {div_yield*100:.1f}%"
                confidence = 0.5
            else:
                direction = "neutral"
                finding = "No dividend"
                evidence = "Stock does not pay a dividend"
                confidence = 0.4

            findings.append(ResearchFinding(
                role=self.name,
                finding=finding,
                direction=direction,
                confidence=round(confidence, 2),
                evidence=evidence,
                data_sources=["yfinance"],
                related_symbol=symbol,
            ))

        # Sector
        sector = info.get("sector")
        if sector:
            findings.append(ResearchFinding(
                role=self.name,
                finding=f"Sector: {sector}",
                direction="neutral",
                confidence=0.5,
                evidence=f"Stock operates in {sector} sector",
                data_sources=["yfinance"],
                related_symbol=symbol,
            ))

        return findings


class NewsSentimentRole:
    """
    News and sentiment analysis role.

    Fetches recent headlines from a news API and computes a simple
    sentiment score based on keyword matching. Supports NewsAPI,
    Finnhub, and Alpha Vantage as providers.
    """

    name = "news_sentiment"

    # Simple keyword-based sentiment lexicon
    BULLISH_KEYWORDS = [
        "upgrade", "buy", "outbeat", "beat", "strong", "growth", "surge",
        "rally", "soar", "jump", "gain", "profit", "record", "high",
        "bullish", "positive", "optimistic", "raise", "raises", "momentum",
        "breakthrough", "partnership", "launch", "expansion", "dividend",
    ]
    BEARISH_KEYWORDS = [
        "downgrade", "sell", "miss", "weak", "decline", "plunge", "crash",
        "drop", "fall", "loss", "low", "bearish", "negative", "pessimistic",
        "cut", "cuts", "concern", "risk", "investigation", "lawsuit",
        "recall", "layoff", "debt", "bankruptcy", "warning",
    ]

    def __init__(self, api_key: str = "", provider: str = "", lookback_days: int = 7,
                 max_headlines: int = 20, api_key_env: str = "", cache_ttl_seconds: int = 1800):
        self.api_key = api_key
        self.provider = provider
        self.lookback_days = lookback_days
        self.max_headlines = max_headlines
        self.api_key_env = api_key_env
        self.cache_ttl_seconds = cache_ttl_seconds
        self._bull_re = re.compile(r"\b(" + "|".join(self.BULLISH_KEYWORDS) + r")\b")
        self._bear_re = re.compile(r"\b(" + "|".join(self.BEARISH_KEYWORDS) + r")\b")

    def analyze(self, snapshot: dict, df: pd.DataFrame) -> list[ResearchFinding]:
        """Analyze news sentiment for the symbol."""
        symbol = snapshot.get("symbol", "")

        if not self.api_key or not self.provider:
            return [ResearchFinding(
                role=self.name,
                finding="News sentiment not configured",
                direction="neutral",
                confidence=0.0,
                evidence="No news API key or provider configured. Set research.news_provider and research.news_api_key_env in config.yaml.",
                data_sources=[],
                related_symbol=symbol,
            )]

        # Fetch news headlines (cached ~30 min: free tiers are rate-limited)
        try:
            headlines = _cache_get("news", f"{self.provider}_{symbol}", self.cache_ttl_seconds)
            if headlines is None:
                headlines = self._fetch_headlines(symbol)
                _cache_put("news", f"{self.provider}_{symbol}", headlines)
        except Exception as e:
            return [ResearchFinding(
                role=self.name,
                finding=f"News fetch failed: {type(e).__name__}",
                direction="neutral",
                confidence=0.0,
                evidence=str(e)[:200],
                data_sources=[self.provider],
                related_symbol=symbol,
            )]

        if not headlines:
            return [ResearchFinding(
                role=self.name,
                finding=f"No recent news found for {symbol}",
                direction="neutral",
                confidence=0.0,
                evidence=f"No headlines from the last {self.lookback_days} days",
                data_sources=[self.provider],
                related_symbol=symbol,
            )]

        # Compute sentiment
        sentiment_score, bullish_count, bearish_count = self._compute_sentiment(headlines)

        if sentiment_score > 0.1:
            direction = "bullish"
            confidence = min(0.8, 0.4 + sentiment_score * 0.5)
        elif sentiment_score < -0.1:
            direction = "bearish"
            confidence = min(0.8, 0.4 + abs(sentiment_score) * 0.5)
        else:
            direction = "neutral"
            confidence = 0.4

        finding_text = f"News sentiment: {direction} ({bullish_count} bullish, {bearish_count} bearish headlines)"
        evidence = f"Analyzed {len(headlines)} headlines from last {self.lookback_days} days. Score: {sentiment_score:.2f}"

        return [ResearchFinding(
            role=self.name,
            finding=finding_text,
            direction=direction,
            confidence=round(confidence, 2),
            evidence=evidence,
            data_sources=[self.provider],
            related_symbol=symbol,
        )]

    def _fetch_headlines(self, symbol: str) -> list[str]:
        """Fetch news headlines from the configured provider."""
        import urllib.request
        import urllib.parse
        import json as json_mod
        from datetime import datetime, timedelta, timezone

        if self.provider == "newsapi":
            return self._fetch_newsapi(symbol)
        elif self.provider == "finnhub":
            return self._fetch_finnhub(symbol)
        elif self.provider == "alpha_vantage":
            return self._fetch_alpha_vantage(symbol)
        else:
            raise ValueError(f"Unknown news provider: {self.provider}")

    def _fetch_newsapi(self, symbol: str) -> list[str]:
        """Fetch headlines from NewsAPI.org."""
        import urllib.request
        import urllib.parse
        import json as json_mod
        from datetime import datetime, timedelta, timezone

        # NewsAPI free tier: everything endpoint, 100 requests/day
        to_date = datetime.now(timezone.utc)
        from_date = to_date - timedelta(days=self.lookback_days)

        query = urllib.parse.quote(f"{symbol} stock")
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={query}&"
            f"from={from_date.strftime('%Y-%m-%d')}&"
            f"to={to_date.strftime('%Y-%m-%d')}&"
            f"language=en&"
            f"sortBy=relevancy&"
            f"pageSize={self.max_headlines}&"
            f"apiKey={self.api_key}"
        )

        req = urllib.request.Request(url, headers={"User-Agent": "finance-bot/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json_mod.loads(resp.read().decode("utf-8"))

        if data.get("status") != "ok":
            raise RuntimeError(f"NewsAPI error: {data.get('message', 'unknown')}")

        articles = data.get("articles", [])
        return [a.get("title", "") for a in articles if a.get("title")]

    def _fetch_finnhub(self, symbol: str) -> list[str]:
        """Fetch headlines from Finnhub."""
        import urllib.request
        import json as json_mod
        from datetime import datetime, timedelta, timezone

        to_date = datetime.now(timezone.utc)
        from_date = to_date - timedelta(days=self.lookback_days)

        url = (
            f"https://finnhub.io/api/v1/company-news?"
            f"symbol={symbol}&"
            f"from={from_date.strftime('%Y-%m-%d')}&"
            f"to={to_date.strftime('%Y-%m-%d')}&"
            f"token={self.api_key}"
        )

        req = urllib.request.Request(url, headers={"User-Agent": "finance-bot/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json_mod.loads(resp.read().decode("utf-8"))

        if not isinstance(data, list):
            raise RuntimeError(f"Finnhub error: unexpected response format")

        return [item.get("headline", "") for item in data[:self.max_headlines] if item.get("headline")]

    def _fetch_alpha_vantage(self, symbol: str) -> list[str]:
        """Fetch headlines from Alpha Vantage."""
        import urllib.request
        import urllib.parse
        import json as json_mod

        query = urllib.parse.quote(symbol)
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=NEWS_SENTIMENT&"
            f"tickers={query}&"
            f"limit={self.max_headlines}&"
            f"apikey={self.api_key}"
        )

        req = urllib.request.Request(url, headers={"User-Agent": "finance-bot/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json_mod.loads(resp.read().decode("utf-8"))

        feed = data.get("feed", [])
        return [item.get("title", "") for item in feed if item.get("title")]

    def _compute_sentiment(self, headlines: list[str]) -> tuple:
        """
        Compute sentiment score from headlines using keyword matching.

        Returns (score, bullish_count, bearish_count) where score is
        in range [-1, 1].
        """
        bullish_count = 0
        bearish_count = 0

        for headline in headlines:
            text = headline.lower()
            # Whole-word matching: substring matching counted "low" in
            # "follow" and "gain" in "against".
            bullish = bool(self._bull_re.search(text))
            bearish = bool(self._bear_re.search(text))
            if bullish:
                bullish_count += 1
            if bearish:
                bearish_count += 1

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0, 0, 0

        score = (bullish_count - bearish_count) / total
        return score, bullish_count, bearish_count


class BullBearRole:
    """
    Bull/Bear analysis role.

    Synthesizes findings from other roles into a bullish and bearish
    thesis with evidence. This is the "devil's advocate" role that
    ensures both sides are considered.
    """

    name = "bull_bear"

    def __init__(self):
        pass

    def analyze(self, snapshot: dict, df: pd.DataFrame, other_findings: list[ResearchFinding] = None) -> list[ResearchFinding]:
        """Synthesize bull/bear thesis from all available findings."""
        findings = []
        symbol = snapshot.get("symbol", "")
        other_findings = other_findings or []

        # Count bullish vs bearish from other roles
        bullish_count = sum(1 for f in other_findings if f.direction == "bullish")
        bearish_count = sum(1 for f in other_findings if f.direction == "bearish")
        total = len(other_findings)

        if total == 0:
            return [ResearchFinding(
                role=self.name,
                finding="Insufficient data for bull/bear analysis",
                direction="neutral",
                confidence=0.0,
                evidence="No other research findings to synthesize",
                data_sources=[],
                related_symbol=symbol,
            )]

        bullish_pct = bullish_count / total
        bearish_pct = bearish_count / total

        # Bull thesis
        if bullish_pct > 0.4:
            findings.append(ResearchFinding(
                role=self.name,
                finding=f"Bull thesis: {bullish_count}/{total} indicators bullish",
                direction="bullish",
                confidence=round(min(0.85, 0.4 + bullish_pct * 0.5), 2),
                evidence=f"Majority of technical/fundamental indicators align bullish ({bullish_pct*100:.0f}%)",
                data_sources=[f.data_sources[0] for f in other_findings if f.data_sources],
                related_symbol=symbol,
            ))

        # Bear thesis
        if bearish_pct > 0.3:
            findings.append(ResearchFinding(
                role=self.name,
                finding=f"Bear thesis: {bearish_count}/{total} indicators bearish",
                direction="bearish",
                confidence=round(min(0.85, 0.4 + bearish_pct * 0.5), 2),
                evidence=f"Significant bearish signals present ({bearish_pct*100:.0f}% of indicators)",
                data_sources=[f.data_sources[0] for f in other_findings if f.data_sources],
                related_symbol=symbol,
            ))

        # If mixed, add a neutral synthesis
        if bullish_pct > 0.3 and bearish_pct > 0.3:
            findings.append(ResearchFinding(
                role=self.name,
                finding="Mixed signals: both bullish and bearish evidence present",
                direction="neutral",
                confidence=0.5,
                evidence=f"Conflicting signals: {bullish_count} bullish vs {bearish_count} bearish. Exercise caution.",
                data_sources=[],
                related_symbol=symbol,
            ))

        return findings


class _RoleSkipped(Exception):
    """Internal: role disabled in config."""


class ResearchAggregator:
    """
    Aggregates findings from all research roles into a single
    structured summary for Jev to consume.
    """

    def __init__(self, config: dict = None):
        config = config or {}
        research_cfg = config.get("research", {})

        # Only the configured roles run (previously all four always ran).
        self.roles = set(research_cfg.get(
            "roles", ["technical", "fundamental", "news_sentiment", "bull_bear"]
        ))
        self.technical = TechnicalRole()
        self.fundamental = FundamentalRole()
        self.news = NewsSentimentRole(
            api_key="",  # loaded from env at call time
            provider=research_cfg.get("news_provider", ""),
            lookback_days=research_cfg.get("news_lookback_days", 7),
            max_headlines=research_cfg.get("news_max_headlines", 20),
            api_key_env=research_cfg.get("news_api_key_env", ""),
        )
        self.bull_bear = BullBearRole()
        self._data_provider = None

    def set_data_provider(self, data_provider):
        """Set the data provider for fundamental analysis."""
        self._data_provider = data_provider
        self.fundamental.data_provider = data_provider

    def run_full_analysis(self, snapshot: dict, df: pd.DataFrame) -> dict:
        """
        Run all research roles and return aggregated findings.

        Returns a dict with:
          - findings: list of all ResearchFinding dicts
          - summary: aggregated direction and confidence
          - by_role: findings grouped by role
        """
        all_findings = []

        # Technical analysis
        try:
            if "technical" not in self.roles:
                raise _RoleSkipped
            tech_findings = self.technical.analyze(snapshot, df)
            all_findings.extend(tech_findings)
        except _RoleSkipped:
            pass
        except Exception as e:
            logger.error("Technical role failed: %s", e)

        # Fundamental analysis
        try:
            if "fundamental" not in self.roles:
                raise _RoleSkipped
            fund_findings = self.fundamental.analyze(snapshot, df)
            all_findings.extend(fund_findings)
        except _RoleSkipped:
            pass
        except Exception as e:
            logger.error("Fundamental role failed: %s", e)

        # News sentiment — load API key from env at call time
        # (previously read a non-existent attribute, fell back to NEWS_API_KEY
        # and the role was permanently "not configured")
        try:
            if "news_sentiment" not in self.roles:
                raise _RoleSkipped
            news_api_key = os.getenv(self.news.api_key_env, "") if self.news.api_key_env else ""
            if not news_api_key and self.news.provider:
                news_api_key = os.getenv("NEWS_API_KEY", "")
            self.news.api_key = news_api_key
            news_findings = self.news.analyze(snapshot, df)
            all_findings.extend(news_findings)
        except _RoleSkipped:
            pass
        except Exception as e:
            logger.error("News sentiment role failed: %s", e)

        # Bull/Bear synthesis
        try:
            if "bull_bear" not in self.roles:
                raise _RoleSkipped
            bb_findings = self.bull_bear.analyze(snapshot, df, all_findings)
            all_findings.extend(bb_findings)
        except _RoleSkipped:
            pass
        except Exception as e:
            logger.error("Bull/Bear role failed: %s", e)

        # Aggregate
        bullish = sum(1 for f in all_findings if f.direction == "bullish")
        bearish = sum(1 for f in all_findings if f.direction == "bearish")
        neutral = sum(1 for f in all_findings if f.direction == "neutral")
        avg_confidence = sum(f.confidence for f in all_findings) / len(all_findings) if all_findings else 0

        if bullish > bearish and bullish > neutral:
            overall_direction = "bullish"
        elif bearish > bullish and bearish > neutral:
            overall_direction = "bearish"
        else:
            overall_direction = "neutral"

        by_role = {}
        for f in all_findings:
            by_role.setdefault(f.role, []).append(f.to_dict())

        return {
            "findings": [f.to_dict() for f in all_findings],
            "summary": {
                "total_findings": len(all_findings),
                "bullish": bullish,
                "bearish": bearish,
                "neutral": neutral,
                "overall_direction": overall_direction,
                "avg_confidence": round(avg_confidence, 2),
            },
            "by_role": by_role,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
