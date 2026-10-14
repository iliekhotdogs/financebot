"""
Snapshot collection — gathers a timestamped market/account/position/
strategy snapshot for a single symbol.

This snapshot is the single source of truth sent to both the advisor
and Jev. No future prices or post-trade outcomes are included.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from data import DataProvider
from risk import RiskManager


def _utc_now_iso() -> str:
    """UTC ISO 8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _snapshot_id(timestamp: str, symbol: str) -> str:
    """Generate a stable, deterministic snapshot ID from timestamp + symbol."""
    raw = f"{timestamp}|{symbol}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_snapshot(
    dp: DataProvider,
    risk_mgr: RiskManager,
    symbol: str,
    strategy_signal: int,
    strategy_name: str,
    df: pd.DataFrame,
    sma_short: int = 20,
    sma_long: int = 50,
    account_state: Optional[dict] = None,
    positions: Optional[list] = None,
) -> dict:
    """
    Collect one timestamped market/account/position/strategy snapshot.

    Parameters
    ----------
    dp : DataProvider
        Data provider for account and position data.
    risk_mgr : RiskManager
        Risk manager for balance info.
    symbol : str
        The symbol to snapshot.
    strategy_signal : int
        The strategy's current signal (-1, 0, 1).
    strategy_name : str
        Human-readable strategy name.
    df : pd.DataFrame
        Historical OHLCV data (already fetched, at point-in-time).
    sma_short, sma_long : int
        SMA windows (for computing current SMA values).
    account_state, positions : optional
        Broker account dict / positions list fetched ONCE per cycle. When
        omitted they are fetched here (2 HTTP calls) — never do that inside
        a per-symbol loop.

    Returns
    -------
    dict
        A snapshot dict with snapshot_id, timestamp, and all fields.
    """
    timestamp = _utc_now_iso()
    sid = _snapshot_id(timestamp, symbol)

    # --- Market data (point-in-time) ---
    latest = df.iloc[-1]
    close_val = float(latest["Close"])

    # Compute SMAs at this point in time
    sma_short_val = None
    sma_long_val = None
    if len(df) >= sma_short:
        sma_short_val = float(df["Close"].rolling(window=sma_short).mean().iloc[-1])
    if len(df) >= sma_long:
        sma_long_val = float(df["Close"].rolling(window=sma_long).mean().iloc[-1])

    market = {
        "close": close_val,
        "open": float(latest.get("Open", close_val)),
        "high": float(latest.get("High", close_val)),
        "low": float(latest.get("Low", close_val)),
        "volume": float(latest.get("Volume", 0)),
        "sma_short": sma_short_val,
        "sma_long": sma_long_val,
        "sma_short_window": sma_short,
        "sma_long_window": sma_long,
        "bar_timestamp": str(latest.name) if latest.name else None,
        "bars_count": len(df),
    }

    # --- Account data (point-in-time) ---
    account = {
        "cash": float(risk_mgr.balance),
        "portfolio_value": float(risk_mgr.balance),
        "currency": "USD",
        "peak_balance": float(risk_mgr.peak_balance),
    }
    # Live Alpaca account state: prefer the per-cycle prefetch
    if account_state is not None or dp.trading_client:
        try:
            live_acct = account_state if account_state is not None else dp.get_account()
            account["cash"] = float(live_acct.get("cash", account["cash"]))
            account["portfolio_value"] = float(
                live_acct.get("portfolio_value", account["portfolio_value"])
            )
            account["currency"] = live_acct.get("currency", "USD")
            account["live"] = True
        except Exception:
            account["live"] = False
    else:
        account["live"] = False

    # --- Position data (point-in-time) ---
    if positions is None:
        positions = []
        if dp.trading_client:
            try:
                positions = dp.get_positions()
            except Exception:
                positions = []

    symbol_pos = None
    for p in positions:
        if p.get("symbol") == symbol:
            symbol_pos = {
                "symbol": p.get("symbol"),
                "qty": float(p.get("qty", 0)),
                "avg_entry_price": float(p.get("avg_entry_price", 0)),
                "unrealized_pl": float(p.get("unrealized_pl", 0)),
            }
            break

    # --- Valid actions ---
    # If we have no position: buy or hold
    # If we have a long position: sell or hold
    # Short signals only make sense if we have a position to close
    has_position = symbol_pos is not None and float(symbol_pos.get("qty", 0)) > 0
    if has_position:
        valid_actions = ["sell", "hold"]
    else:
        valid_actions = ["buy", "hold"]

    return {
        "snapshot_id": sid,
        "timestamp": timestamp,
        "symbol": symbol,
        "market": market,
        "account": account,
        "position": symbol_pos,
        "strategy_signal": strategy_signal,
        "strategy_name": strategy_name,
        "valid_actions": valid_actions,
    }


def snapshot_to_text(snap: dict) -> str:
    """Render a snapshot dict as human-readable text for model consumption."""
    m = snap["market"]
    pos = snap.get("position")
    acct = snap["account"]
    lines = [
        f"Timestamp (UTC): {snap['timestamp']}",
        f"Snapshot ID: {snap['snapshot_id']}",
        f"Symbol: {snap['symbol']}",
        f"Strategy: {snap['strategy_name']}",
        f"Signal: {snap['strategy_signal']} (1=buy, -1=sell, 0=hold)",
        f"Valid actions: {snap['valid_actions']}",
        f"Latest close: {m.get('close')}",
        f"SMA short ({m.get('sma_short_window')}): {m.get('sma_short')}",
        f"SMA long ({m.get('sma_long_window')}): {m.get('sma_long')}",
        f"Cash: {acct.get('cash')}",
        f"Portfolio value: {acct.get('portfolio_value')}",
        f"Current position: {pos['symbol'] if pos else 'none'}",
    ]
    if pos:
        lines.append(f"  Position qty: {pos.get('qty')}")
        lines.append(f"  Avg entry: {pos.get('avg_entry_price')}")
        lines.append(f"  Unrealized P/L: {pos.get('unrealized_pl')}")
    return "\n".join(lines)
