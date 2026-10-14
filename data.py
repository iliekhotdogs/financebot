"""
Data integration layer: Alpaca (real-time + trading) + yfinance (historical/backfill).

Design:
  - yfinance: fetch historical bars, splits, dividends, fundamentals for backfilling
    and baseline data. Free but fragile (unofficial scraping).
  - Alpaca: real-time bars/quotes, order execution, account/portfolio state.
    Authoritative for live trading in paper mode.

The DataProvider class abstracts both so the strategy layer gets unified access.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

# yfinance for historical/backfill
import yfinance as yf

# Alpaca for real-time data + trading
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data import DataFeed
from alpaca.data.enums import Adjustment
from alpaca.trading.requests import (
    MarketOrderRequest, StopLossRequest, GetOrdersRequest, ReplaceOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus

logger = logging.getLogger(__name__)


class DataProvider:
    """Unified data provider backed by Alpaca (live) and yfinance (historical)."""

    def __init__(
        self,
        alpaca_key: Optional[str] = None,
        alpaca_secret: Optional[str] = None,
        alpaca_base_url: str = "https://paper-api.alpaca.markets",
        paper: bool = True,
    ):
        self.paper = paper
        self.alpaca_key = alpaca_key or os.getenv("APCA_API_KEY_ID")
        self.alpaca_secret = alpaca_secret or os.getenv("APCA_API_SECRET_KEY")
        self.alpaca_base_url = alpaca_base_url or os.getenv(
            "APCA_API_BASE_URL", "https://paper-api.alpaca.markets"
        )

        # Alpaca trading client (for orders / account)
        self.trading_client: Optional[TradingClient] = None
        # Alpaca market data client (for real-time bars/quotes)
        self.data_client: Optional[StockHistoricalDataClient] = None

        if self.alpaca_key and self.alpaca_secret:
            self._init_alpaca()
        else:
            logger.warning(
                "Alpaca credentials not found. Only yfinance data will be available. "
                "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your environment or .env"
            )

    def _init_alpaca(self):
        """Initialize Alpaca trading + data clients."""
        try:
            self.trading_client = TradingClient(
                api_key=self.alpaca_key,
                secret_key=self.alpaca_secret,
                paper=self.paper,
            )
            self.data_client = StockHistoricalDataClient(
                api_key=self.alpaca_key,
                secret_key=self.alpaca_secret,
            )
            logger.info("Alpaca clients initialized (paper=%s)", self.paper)
        except Exception as e:
            logger.error("Failed to initialize Alpaca client: %s", e)
            raise

    # ------------------------------------------------------------------ #
    # yfinance — historical / backfill
    # ------------------------------------------------------------------ #

    def fetch_yf_history(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from yfinance (backfill / research)."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            logger.info(
                "yfinance: fetched %s %s/%s — %d rows", symbol, period, interval, len(df)
            )
            return df
        except Exception as e:
            logger.error("yfinance fetch failed for %s: %s", symbol, e)
            return pd.DataFrame()

    def fetch_yf_fundamentals(self, symbol: str) -> dict:
        """Fetch fundamental data (info) from yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error("yfinance fundamentals fetch failed for %s: %s", symbol, e)
            return {}

    def fetch_yf_dividends_splits(self, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch dividends and splits from yfinance for backfill."""
        try:
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends
            splits = ticker.splits
            return dividends, splits
        except Exception as e:
            logger.error("yfinance dividends/splits failed for %s: %s", symbol, e)
            return pd.DataFrame(), pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Alpaca — real-time data
    # ------------------------------------------------------------------ #

    def get_latest_quote(self, symbol: str) -> dict:
        """Get the latest real-time quote from Alpaca."""
        if not self.data_client:
            raise RuntimeError("Alpaca data client not initialized")

        request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        quote = self.data_client.get_stock_latest_quote(request)
        return quote[symbol].dict()

    def get_latest_trade(self, symbol: str) -> dict:
        """Get the latest real-time trade from Alpaca."""
        if not self.data_client:
            raise RuntimeError("Alpaca data client not initialized")

        request = StockLatestTradeRequest(symbol_or_symbols=[symbol])
        trade = self.data_client.get_stock_latest_trade(request)
        return trade[symbol].dict()

    def fetch_alpaca_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.Day,
    ) -> pd.DataFrame:
        """Fetch historical bars from Alpaca (authoritative real-time source)."""
        if not self.data_client:
            raise RuntimeError("Alpaca data client not initialized")

        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe,
            start=start,
            end=end,
            feed=DataFeed.IEX,
            # Alpaca's default is RAW: a 10:1 split (NVDA/AVGO 2024) reads as a
            # -90% crash to the SMAs, ATR and the learning backtests.
            adjustment=Adjustment.ALL,
        )
        df = self.data_client.get_stock_bars(request).df
        if symbol in df.columns.get_level_values(0):
            df = df.loc[symbol]
        # Normalize column names to match yfinance format (capitalized)
        if not df.empty:
            rename_map = {
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume", "vwap": "Vwap",
                "trade_count": "Trade Count", "timestamp": "Timestamp",
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        return df

    def fetch_alpaca_bars_multi(
        self,
        symbols: list,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.Day,
        chunk_size: int = 100,
    ) -> dict:
        """Fetch bars for many symbols in batched requests.

        Returns {symbol: DataFrame} with yfinance-style column names. One
        request per `chunk_size` symbols instead of one per symbol.
        """
        if not self.data_client:
            raise RuntimeError("Alpaca data client not initialized")

        out = {}
        rename_map = {
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume", "vwap": "Vwap",
            "trade_count": "Trade Count",
        }
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=chunk, timeframe=timeframe,
                    start=start, end=end, feed=DataFeed.IEX, adjustment=Adjustment.ALL,
                )
                df = self.data_client.get_stock_bars(request).df
            except Exception as e:
                logger.warning("Batched Alpaca bars failed for %d symbols: %s", len(chunk), e)
                continue
            if df.empty:
                continue
            for sym in df.index.get_level_values(0).unique():
                sdf = df.loc[sym].rename(columns=rename_map)
                out[sym] = sdf
        return out

    def get_snapshots(self, symbols: list, chunk_size: int = 500) -> dict:
        """Latest daily bar / quote snapshot for many symbols (batched).

        Returns {symbol: {"price": float, "volume": float, "prev_volume": float}}.
        Used by the dynamic-universe liquidity screen.
        """
        if not self.data_client:
            raise RuntimeError("Alpaca data client not initialized")
        out = {}
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            try:
                snaps = self.data_client.get_stock_snapshot(
                    StockSnapshotRequest(symbol_or_symbols=chunk, feed=DataFeed.IEX)
                )
            except Exception as e:
                logger.warning("Snapshot request failed for %d symbols: %s", len(chunk), e)
                continue
            for sym, snap in snaps.items():
                daily = getattr(snap, "daily_bar", None)
                prev = getattr(snap, "previous_daily_bar", None)
                price = None
                if daily is not None:
                    price = float(daily.close)
                elif getattr(snap, "latest_trade", None) is not None:
                    price = float(snap.latest_trade.price)
                if price is None:
                    continue
                out[sym] = {
                    "price": price,
                    "volume": float(daily.volume) if daily is not None else 0.0,
                    "prev_volume": float(prev.volume) if prev is not None else 0.0,
                    "prev_close": float(prev.close) if prev is not None else price,
                }
        return out

    # ------------------------------------------------------------------ #
    # Hybrid: backfill gap with yfinance, authoritative live from Alpaca
    # ------------------------------------------------------------------ #

    def get_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.Day,
    ) -> pd.DataFrame:
        """
        Fetch complete history for [start, end].
        Uses Alpaca (real-time, authoritative) and fills any gaps with yfinance backfill.
        """
        # Try Alpaca first
        if self.data_client:
            try:
                alpaca_df = self.fetch_alpaca_bars(symbol, start, end, timeframe)
                if not alpaca_df.empty:
                    # If Alpaca covers the full range, return it
                    if alpaca_df.index[0] <= start + timedelta(days=1) and \
                       alpaca_df.index[-1] >= end - timedelta(days=1):
                        return alpaca_df
                    # Otherwise, backfill the gap with yfinance
                    yf_df = self._backfill_yf(symbol, start, end, timeframe)
                    if not yf_df.empty:
                        combined = pd.concat([alpaca_df, yf_df]).drop_duplicates()
                        combined = combined.sort_index()
                        return combined
                return alpaca_df
            except Exception as e:
                logger.warning("Alpaca bars failed for %s, falling back to yfinance: %s", symbol, e)

        # Pure yfinance fallback (no Alpaca configured)
        return self._backfill_yf(symbol, start, end, timeframe)

    def _backfill_yf(self, symbol: str, start: datetime, end: datetime, timeframe: TimeFrame) -> pd.DataFrame:
        """Backfill using yfinance with appropriate period/interval mapping."""
        yf_interval_map = {
            TimeFrame.Minute: "2y",
            TimeFrame.Hour: "2y",
            TimeFrame.Day: "max",
            TimeFrame.Week: "max",
            TimeFrame.Month: "max",
        }
        yf_interval_str = {
            TimeFrame.Minute: "2m",
            TimeFrame.Hour: "1h",
            TimeFrame.Day: "1d",
            TimeFrame.Week: "1wk",
            TimeFrame.Month: "1mo",
        }.get(timeframe, "1d")

        period = yf_interval_map.get(timeframe, "1y")
        df = self.fetch_yf_history(symbol, period=period, interval=yf_interval_str)

        if df.empty:
            return df

        # Filter to requested date range (handle tz-aware vs naive datetimes)
        if df.index.tz is not None:
            start = pd.Timestamp(start).tz_localize(df.index.tz) if pd.Timestamp(start).tz is None else pd.Timestamp(start)
            end = pd.Timestamp(end).tz_localize(df.index.tz) if pd.Timestamp(end).tz is None else pd.Timestamp(end)
        mask = (df.index >= start) & (df.index <= end)
        return df.loc[mask]

    # ------------------------------------------------------------------ #
    # Alpaca — trading / account
    # ------------------------------------------------------------------ #

    def get_account(self) -> dict:
        """Get Alpaca account state."""
        if not self.trading_client:
            raise RuntimeError("Alpaca trading client not initialized")
        return self.trading_client.get_account().__dict__

    def get_positions(self) -> list:
        """Get current open positions from Alpaca."""
        if not self.trading_client:
            raise RuntimeError("Alpaca trading client not initialized")
        positions = self.trading_client.get_all_positions()
        return [p.__dict__ for p in positions]

    def get_clock(self) -> dict:
        """Check if market is open."""
        if not self.trading_client:
            raise RuntimeError("Alpaca trading client not initialized")
        return self.trading_client.get_clock().__dict__

    def get_all_tradeable_symbols(self) -> list:
        """Get all tradeable symbols from Alpaca.
        
        Returns a sorted list of all tradable asset symbols available
        on the Alpaca platform. This enables the bot to trade any stock
        without a hardcoded universe.
        """
        if not self.trading_client:
            raise RuntimeError("Alpaca trading client not initialized")
        assets = self.trading_client.get_all_assets()
        return sorted([a.symbol for a in assets if a.tradable])

    def get_all_assets_meta(self) -> list:
        """Get metadata for all Alpaca assets in ONE request.

        Returns a list of dicts with symbol, asset_class, tradable, status.
        Use this instead of looping get_asset_info() per symbol — that is
        one HTTP request per symbol (~13.5k calls) and stalls a cycle.
        """
        if not self.trading_client:
            raise RuntimeError("Alpaca trading client not initialized")
        out = []
        for a in self.trading_client.get_all_assets():
            out.append({
                "symbol": getattr(a, "symbol", None),
                "asset_class": _enum_value(getattr(a, "asset_class", None)),
                "tradable": bool(getattr(a, "tradable", False)),
                "status": _enum_value(getattr(a, "status", None)),
                "exchange": _enum_value(getattr(a, "exchange", None)),
                "fractionable": bool(getattr(a, "fractionable", False)),
                "name": getattr(a, "name", "") or "",
            })
        return out

    def get_asset_info(self, symbol: str) -> dict:
        """Get asset info from Alpaca."""
        if not self.trading_client:
            raise RuntimeError("Alpaca trading client not initialized")
        asset = self.trading_client.get_asset(symbol)
        return asset.__dict__ if asset else {}

    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        """Open (unfilled / working) orders, optionally for one symbol."""
        if not self.trading_client:
            raise RuntimeError("Alpaca trading client not initialized")
        req = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol] if symbol else None,
            nested=True,
        )
        return [_order_to_dict(o) for o in self.trading_client.get_orders(filter=req)]

    def cancel_open_orders(self, symbol: str) -> int:
        """Cancel every open order for `symbol` (e.g. a protective stop
        that would otherwise lock the shares we want to sell)."""
        n = 0
        for o in self.get_open_orders(symbol):
            try:
                self.trading_client.cancel_order_by_id(o["id"])
                n += 1
            except Exception as e:
                logger.warning("Cancel order %s for %s failed: %s", o["id"], symbol, e)
        return n

    def replace_stop(self, order_id: str, stop_price: float) -> dict:
        """Ratchet an existing stop order to a new stop price."""
        if not self.trading_client:
            raise RuntimeError("Alpaca trading client not initialized")
        order = self.trading_client.replace_order_by_id(
            order_id, ReplaceOrderRequest(stop_price=round(stop_price, 2))
        )
        return _order_to_dict(order)

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str = "buy",
        time_in_force: str = "day",
        stop_loss_price: Optional[float] = None,
    ) -> dict:
        """
        Place a market order via Alpaca (paper only).

        alpaca-py >= 0.30 takes a request object: submit_order(order_data=...).
        Passing symbol=/qty=/side= keywords raises TypeError, which is why no
        order had ever been placed before this fix.

        If `stop_loss_price` is given for a buy, the entry is submitted as an
        OTO order with a broker-side GTC stop, so the position stays protected
        even when the bot is not running.
        """
        if not self.trading_client:
            raise RuntimeError("Alpaca trading client not initialized")

        # Confirm we are in paper mode
        if not self.paper:
            raise RuntimeError("Live trading is NOT enabled. Set APCA_PAPER=true for paper trading.")

        qty = float(qty)
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        kwargs = dict(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce(time_in_force.lower()),
        )
        if stop_loss_price is not None and order_side == OrderSide.BUY:
            # OTO legs must be whole shares; GTC so the stop survives overnight.
            kwargs["qty"] = float(int(qty))
            kwargs["time_in_force"] = TimeInForce.GTC
            kwargs["order_class"] = OrderClass.OTO
            kwargs["stop_loss"] = StopLossRequest(stop_price=round(float(stop_loss_price), 2))

        logger.warning(
            "Placing ORDER: %s %s %s stop=%s (paper=%s)",
            side, kwargs["qty"], symbol, stop_loss_price, self.paper,
        )
        order = self.trading_client.submit_order(order_data=MarketOrderRequest(**kwargs))
        return _order_to_dict(order)

    def close_position(self, symbol: str) -> dict:
        """Cancel any working orders for `symbol` and liquidate the position."""
        if not self.trading_client:
            raise RuntimeError("Alpaca trading client not initialized")
        if not self.paper:
            raise RuntimeError("Live trading is NOT enabled. Set APCA_PAPER=true for paper trading.")
        self.cancel_open_orders(symbol)
        order = self.trading_client.close_position(symbol)
        return _order_to_dict(order)


def _enum_value(v):
    """alpaca-py returns enums for asset_class/status/etc.; compare on .value."""
    return getattr(v, "value", v)


def _order_to_dict(order) -> dict:
    """Flatten an alpaca Order model into plain types (ids as str)."""
    if isinstance(order, dict):
        return order
    d = dict(order.__dict__)
    for k in ("id", "client_order_id"):
        if d.get(k) is not None:
            d[k] = str(d[k])
    for k in ("side", "status", "type", "order_class", "time_in_force"):
        if k in d:
            d[k] = _enum_value(d[k])
    legs = d.get("legs") or []
    d["legs"] = [_order_to_dict(l) for l in legs]
    return d
