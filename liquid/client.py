"""Liquid Trade client — stub for Phase 2 live trading.

This module provides a unified interface to Liquid Trade perps via the
Co-Invest Computer MCP server. It is DISABLED until paper trading with
Alpaca + yfinance is validated and the operator approves live trading.

Usage (after approval):
    from liquid.client import LiquidClient
    client = LiquidClient()  # uses Hermes MCP runtime
    client.status()          # check paper mode
    client.portfolio()       # read portfolio
    client.place_order(...)  # submit order (requires approval)

The MCP tools are invoked through the Hermes runtime, not directly.
This client must be called from within a Hermes agent context or a
script triggered by a Hermes cron job with MCP toolsets enabled.
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

LIQUID_MCP_URL = "https://coinvest-computer.liquid.trade/mcp"
DEFAULT_SYMBOLS = ["NVDA/USD perp", "AAPL/USD perp", "TSLA/USD perp"]


class LiquidClient:
    """
    Client for Liquid Trade perps via Co-Invest MCP.

    DISABLED by default. Check `enabled` before any calls.
    Paper mode is enforced — no live orders without operator approval.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.mcp_url = os.getenv(
            "LIQUID_MCP_URL", LIQUID_MCP_URL
        )
        self._paper = True  # always paper until operator flips live

    @property
    def paper(self) -> bool:
        """Returns True if in paper mode (always, until overridden)."""
        return self._paper

    def status(self) -> dict:
        """Check paper trading status and market availability."""
        if not self.enabled:
            return {
                "enabled": False,
                "paper": True,
                "status": "disabled — paper trading via Alpaca + yfinance active",
                "message": "Liquid integration not yet activated. "
                "Paper trading with Alpaca + yfinance is the current mode.",
            }
        # Real MCP calls would go here via Hermes runtime
        return {
            "enabled": True,
            "paper": self._paper,
            "status": "active (paper)" if self._paper else "active (LIVE)",
        }

    def portfolio(self) -> dict:
        """Get Liquid portfolio (paper balance, positions)."""
        if not self.enabled:
            raise RuntimeError(
                "Liquid client not enabled. Set enabled=True only after "
                "paper trading validation and operator approval."
            )
        # MCP call: get_portfolio
        raise NotImplementedError(
            "Liquid MCP integration requires Hermes agent context. "
            "Use a cron job with MCP toolsets enabled."
        )

    def market_overview(self) -> dict:
        """Get market overview for configured symbols."""
        if not self.enabled:
            return {
                "enabled": False,
                "symbols": [],
                "message": "Liquid market data not active in paper mode.",
            }
        # MCP call: show_market_overview
        raise NotImplementedError("Requires Hermes MCP runtime context.")

    def place_order(self, symbol: str, qty: float, side: str = "buy",
                    leverage: float = 1.0,
                    risk_manager=None) -> dict:
        """
        Place an order through Liquid Trade.

        REQUIRES:
        1. Liquid client enabled
        2. Paper trading mode confirmed
        3. RiskManager approval (pre-trade check)
        4. Explicit operator approval for first live trade
        """
        if not self.enabled:
            raise RuntimeError("Liquid trading not enabled.")
        if not self._paper:
            raise RuntimeError(
                "LIVE trading requires explicit operator approval. "
                "Set APCA_PAPER=false and confirm with operator."
            )

        # Risk check
        if risk_manager:
            check = risk_manager.check_max_leverage(leverage)
            if not check.passed:
                return {"status": "rejected", "reason": check.reason}

        # MCP call: execute_order — requires Hermes MCP runtime
        raise NotImplementedError(
            "Order execution via Liquid MCP requires a Hermes agent cron job "
            "with MCP toolsets enabled. Define a separate cron job that calls "
            "execute_order through the MCP tools."
        )
