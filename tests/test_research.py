"""Tests for research roles."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BOT_DIR)

from research.roles import (
    ResearchFinding, TechnicalRole, FundamentalRole,
    NewsSentimentRole, BullBearRole, ResearchAggregator,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    df = pd.DataFrame({
        "Open": close + np.random.randn(100) * 0.1,
        "High": close + abs(np.random.randn(100) * 0.5),
        "Low": close - abs(np.random.randn(100) * 0.5),
        "Close": close,
        "Volume": np.random.randint(1000000, 5000000, 100),
    }, index=dates)
    return df


@pytest.fixture
def sample_snapshot():
    return {
        "snapshot_id": "test-snap-001",
        "timestamp": "2026-10-01T00:00:00+00:00",
        "symbol": "AAPL",
        "market": {
            "close": 150.0,
            "open": 149.0,
            "high": 151.0,
            "low": 148.0,
            "volume": 1000000,
            "sma_short": 148.0,
            "sma_long": 145.0,
            "sma_short_window": 20,
            "sma_long_window": 50,
        },
        "account": {"cash": 100000, "portfolio_value": 100000, "currency": "USD"},
        "position": None,
        "strategy_signal": 1,
        "strategy_name": "SMA Cross (20/50)",
        "valid_actions": ["buy", "hold"],
    }


class TestResearchFinding:
    def test_create_finding(self):
        f = ResearchFinding(
            role="technical",
            finding="Test finding",
            direction="bullish",
            confidence=0.75,
            evidence="Test evidence",
        )
        assert f.role == "technical"
        assert f.direction == "bullish"
        assert f.confidence == 0.75
        assert f.timestamp != ""

    def test_to_dict(self):
        f = ResearchFinding(
            role="technical",
            finding="Test",
            direction="neutral",
            confidence=0.5,
            evidence="Test evidence",
        )
        d = f.to_dict()
        assert d["role"] == "technical"
        assert d["direction"] == "neutral"


class TestTechnicalRole:
    def test_analyze_returns_findings(self, sample_snapshot, sample_df):
        role = TechnicalRole()
        findings = role.analyze(sample_snapshot, sample_df)
        assert len(findings) > 0
        assert all(isinstance(f, ResearchFinding) for f in findings)

    def test_analyze_insufficient_data(self, sample_snapshot):
        role = TechnicalRole()
        empty_df = pd.DataFrame()
        findings = role.analyze(sample_snapshot, empty_df)
        assert len(findings) == 1
        assert findings[0].confidence == 0.0

    def test_rsi_computation(self, sample_df):
        role = TechnicalRole()
        rsi = role._compute_rsi(sample_df)
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_macd_computation(self, sample_df):
        role = TechnicalRole()
        macd, signal, hist = role._compute_macd(sample_df)
        assert macd is not None
        assert signal is not None

    def test_atr_computation(self, sample_df):
        role = TechnicalRole()
        atr = role._compute_atr(sample_df)
        assert atr is not None
        assert atr > 0

    def test_support_resistance(self, sample_df):
        role = TechnicalRole()
        support, resistance = role._compute_support_resistance(sample_df)
        assert support is not None
        assert resistance is not None
        assert support < resistance


class TestFundamentalRole:
    def test_analyze_no_provider(self, sample_snapshot, sample_df):
        role = FundamentalRole(data_provider=None)
        findings = role.analyze(sample_snapshot, sample_df)
        assert len(findings) == 1
        assert "not configured" in findings[0].finding.lower()


class TestNewsSentimentRole:
    def test_analyze_not_configured(self, sample_snapshot, sample_df):
        role = NewsSentimentRole(api_key="", provider="")
        findings = role.analyze(sample_snapshot, sample_df)
        assert len(findings) == 1
        assert "not configured" in findings[0].finding.lower()


class TestBullBearRole:
    def test_analyze_no_findings(self, sample_snapshot, sample_df):
        role = BullBearRole()
        findings = role.analyze(sample_snapshot, sample_df, [])
        assert len(findings) == 1
        assert "insufficient" in findings[0].finding.lower()

    def test_analyze_with_findings(self, sample_snapshot, sample_df):
        role = BullBearRole()
        other_findings = [
            ResearchFinding(role="technical", finding="Bullish", direction="bullish", confidence=0.7, evidence="e"),
            ResearchFinding(role="technical", finding="Bullish 2", direction="bullish", confidence=0.6, evidence="e"),
            ResearchFinding(role="technical", finding="Bearish", direction="bearish", confidence=0.5, evidence="e"),
        ]
        findings = role.analyze(sample_snapshot, sample_df, other_findings)
        assert len(findings) >= 1
        # Should have a bull thesis since bullish > bearish
        assert any(f.direction == "bullish" for f in findings)


class TestResearchAggregator:
    def test_run_full_analysis(self, sample_snapshot, sample_df):
        config = {"research": {"enabled": True, "roles": ["technical", "fundamental", "bull_bear"]}}
        agg = ResearchAggregator(config)
        result = agg.run_full_analysis(sample_snapshot, sample_df)
        assert "findings" in result
        assert "summary" in result
        assert "by_role" in result
        assert result["summary"]["total_findings"] > 0

    def test_summary_counts(self, sample_snapshot, sample_df):
        config = {"research": {"enabled": True, "roles": ["technical"]}}
        agg = ResearchAggregator(config)
        result = agg.run_full_analysis(sample_snapshot, sample_df)
        summary = result["summary"]
        assert summary["bullish"] + summary["bearish"] + summary["neutral"] == summary["total_findings"]
