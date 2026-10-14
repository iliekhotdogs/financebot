"""Liquid Trade integration module.

Phase 1 (paper):  Alpaca + yfinance for data + paper execution.
Phase 2 (live):   Liquid Trade perps via Co-Invest MCP.

The Liquid client is a thin wrapper around the Liquid Co-Invest Computer MCP
server (https://coinvest-computer.liquid.trade/mcp). It is DISABLED by
default and can only be activated with explicit operator approval and a
paper-trading gate pass.
"""
from .client import LiquidClient

__all__ = ["LiquidClient"]
