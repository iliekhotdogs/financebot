"""
Strategy v1.1 — SMA Cross with 65% Profit Floor and Runner Hug (ATR trailing).

Extends v1.0 (SMA Crossover entry) with two exit rules:

1. **Hard Profit Floor (The 65% Rule)**:
   - Once a trade gains +2.5%, the stop ratchets to lock in 50% of profit.
   - Once a trade gains +5.0%, the stop locks in 65% of total gain.
   - Mathematically impossible for a big winner to bleed back to break-even.

2. **The Runner Hug (ATR trailing stops tighten as price rallies)**:
   - Initial stop: 3.0× ATR(14) below entry (gives room for normal volatility).
   - As profit grows > +2.5%: trailing distance tightens to 0.8× ATR.
   - As profit grows > +5.0%: trailing distance tightens to 0.6× ATR.
   - Exit when price closes below the trailing stop.

Entry: same as v1.0 — 20 SMA crosses above 50 SMA (long only).
Exit: whichever comes first — SMA crossover exit, 65% Rule floor, or ATR trailing stop.
"""
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """A single completed or open trade."""
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    qty: float = 1.0
    entry_atr: float = 0.0
    stop_price: float = 0.0
    max_gain_pct: float = 0.0
    status: str = "open"  # 'open' | 'closed'

    @property
    def exit_reason(self) -> str:
        if self.status != "closed":
            return ""
        # Determined during backtest
        return self._exit_reason or ""

    @exit_reason.setter
    def exit_reason(self, value: str):
        self._exit_reason = value

    def pnl_pct(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price * 100


def metrics_from_daily(daily_returns: pd.Series) -> dict:
    """Compounded return, max drawdown (negative %), annualised Sharpe."""
    dr = pd.Series(daily_returns).fillna(0.0)
    if len(dr) == 0:
        return {"total_return_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0}
    equity = (1 + dr).cumprod()
    total = (equity.iloc[-1] - 1) * 100
    dd = (equity / equity.cummax() - 1).min() * 100
    sd = dr.std()
    sharpe = float(dr.mean() / sd * np.sqrt(252)) if len(dr) > 1 and sd > 0 else 0.0
    return {"total_return_pct": round(float(total), 2),
            "max_drawdown_pct": round(float(dd), 2),
            "sharpe_ratio": round(sharpe, 2)}


def window_metrics(raw: dict, start) -> dict:
    """
    Restrict backtest output to bars at/after `start`.

    The backtest is run on [warm-up history + window] so indicators are
    defined at the window start; only daily returns in the window and trades
    ENTERED in the window are scored. No look-ahead: data ends at window end.
    """
    out = dict(raw)
    daily = raw.get("daily_returns")
    if daily is None:
        return out
    daily = daily[daily.index >= start]
    out.update(metrics_from_daily(daily))
    out["daily_returns"] = daily
    if "trades" in raw:
        trades = [t for t in raw["trades"] if t["entry_date"] >= start]
        rets = [t["pnl_pct"] for t in trades]
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]
        out.update({
            "trades": trades,
            "trade_returns": rets,
            "total_trades": len(trades),
            "win_rate": round(len(wins) / len(rets) * 100, 1) if rets else 0,
            "avg_win_pct": round(float(np.mean(wins)), 2) if wins else 0,
            "avg_loss_pct": round(float(np.mean(losses)), 2) if losses else 0,
            "total_wins": len(wins),
            "total_losses": len(losses),
        })
    return out


class SMACrossV1_1:
    """
    SMA Cross v1.1 with 65% Rule + Runner Hug.

    Parameters
    ----------
    sma_short : int   — Short SMA window (default 20)
    sma_long : int    — Long SMA window (default 50)
    atr_period : int  — ATR lookback window (default 14)
    atr_mult_initial : float — Initial stop distance in ATR multiples (default 3.0)
    atr_mult_tight : float   — Tightened trailing distance (default 0.8)
    profit_floor_1 : float   — First profit floor threshold (default 0.025 = 2.5%)
    profit_lock_1 : float    — Lock-in at first threshold (default 0.50 = 50%)
    profit_floor_2 : float   — Second profit floor threshold (default 0.05 = 5.0%)
    profit_lock_2 : float    — Lock-in at second threshold (default 0.65 = 65%)
    """

    name = "SMA Cross v1.1 (65% Rule + Runner Hug)"
    version = "v1.1"
    changelog = "v1.1: Added 65% profit floor (lock 50% at +2.5%, 65% at +5.0%) + ATR trailing (Runner Hug). Exit: crossover, profit floor, or trailing stop."

    def __init__(
        self,
        sma_short: int = 20,
        sma_long: int = 50,
        atr_period: int = 14,
        atr_mult_initial: float = 3.0,
        atr_mult_tight: float = 0.8,
        atr_mult_tightest: float = 0.6,
        profit_floor_1: float = 0.025,
        profit_lock_1: float = 0.50,
        profit_floor_2: float = 0.05,
        profit_lock_2: float = 0.65,
    ):
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.atr_period = atr_period
        self.atr_mult_initial = atr_mult_initial
        self.atr_mult_tight = atr_mult_tight
        self.atr_mult_tightest = atr_mult_tightest
        self.profit_floor_1 = profit_floor_1
        self.profit_lock_1 = profit_lock_1
        self.profit_floor_2 = profit_floor_2
        self.profit_lock_2 = profit_lock_2

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """Compute ATR(14)."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    def _compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMA, ATR, and signal columns to the DataFrame."""
        df = df.copy()
        df["sma_short"] = df["Close"].rolling(self.sma_short).mean()
        df["sma_long"] = df["Close"].rolling(self.sma_long).mean()
        df["atr"] = self._compute_atr(df)
        df["signal"] = np.where(df["sma_short"] > df["sma_long"], 1, 0)
        df["position"] = df["signal"].diff()
        return df

    def generate_signal(self, df: pd.DataFrame) -> int:
        """Generate a position signal from the latest data (same interface as v1.0)."""
        if len(df) < self.sma_long:
            return 0
        df = self._compute_signals(df)
        latest = df.iloc[-1]
        if pd.isna(latest["position"]):
            return 0
        if latest["position"] == 1:
            return 1
        elif latest["position"] == -1:
            return -1
        else:
            return int(latest["signal"])

    # ------------------------------------------------------------------ #
    # Live-trading helpers
    #
    # generate_signal() returns the *trend state* (1 on every bar where
    # SMA_short > SMA_long, -1 only on the exact cross-down bar). The
    # backtest only enters on a fresh crossover and exits via the v1.1
    # rules, so live trading must use the same semantics or live and
    # backtest results are not comparable.
    # ------------------------------------------------------------------ #
    @property
    def min_bars(self) -> int:
        """Bars needed before any signal is defined."""
        return max(self.sma_long, self.atr_period) + 2

    def live_signal(self, df: pd.DataFrame, entry_lookback_bars: int = 3) -> int:
        """
        Live signal matching the backtest:
          1  — SMA_short crossed above SMA_long within the last
               `entry_lookback_bars` bars and is still above (fresh entry;
               the lookback tolerates missed cron cycles / days)
         -1  — SMA_short is below SMA_long (trend exit for an open long)
          0  — otherwise
        """
        if len(df) < self.min_bars:
            return 0
        d = self._compute_signals(df)
        latest = d.iloc[-1]
        if pd.isna(latest["sma_long"]):
            return 0
        if latest["sma_short"] < latest["sma_long"]:
            return -1
        recent = d["position"].iloc[-entry_lookback_bars:]
        if (recent == 1).any() and latest["signal"] == 1:
            return 1
        return 0

    def initial_stop(self, df: pd.DataFrame) -> Optional[float]:
        """Initial protective stop: close - atr_mult_initial x ATR."""
        atr = self._compute_atr(df).iloc[-1]
        if pd.isna(atr) or atr <= 0:
            return None
        return float(df["Close"].iloc[-1] - self.atr_mult_initial * atr)

    def evaluate_exit(self, df: pd.DataFrame, entry_price: float,
                      current_stop: Optional[float]) -> dict:
        """
        Apply the v1.1 exit rules (SMA cross-down, 65% profit floor,
        Runner Hug ATR trail) to an open long, using the latest bar.

        Returns {"exit": bool, "reason": str, "stop": float}. The stop
        only ever ratchets up.
        """
        d = self._compute_signals(df)
        row = d.iloc[-1]
        close = float(row["Close"])
        atr = float(row["atr"]) if not pd.isna(row["atr"]) else 0.0
        stop = float(current_stop) if current_stop else (
            entry_price - self.atr_mult_initial * atr if atr > 0 else entry_price * 0.9
        )

        if not pd.isna(row["sma_long"]) and row["sma_short"] < row["sma_long"]:
            return {"exit": True, "reason": "sma_crossover", "stop": stop}

        gain = (close - entry_price) / entry_price if entry_price > 0 else 0.0
        if gain >= self.profit_floor_2:
            stop = max(stop, entry_price * (1 + gain * self.profit_lock_2))
            mult = self.atr_mult_tightest
        elif gain >= self.profit_floor_1:
            stop = max(stop, entry_price * (1 + gain * self.profit_lock_1))
            mult = self.atr_mult_tight
        else:
            mult = self.atr_mult_initial
        if atr > 0:
            stop = max(stop, close - mult * atr)

        if close <= stop:
            return {"exit": True, "reason": f"stop_hit_{mult}x_atr", "stop": stop}
        return {"exit": False, "reason": "", "stop": stop}

    def backtest(self, df: pd.DataFrame) -> dict:
        """
        Run backtest with 65% Rule + Runner Hug exits.

        Exit logic per open position:
        1. SMA crossover (short crosses below long) — original v1.0 exit
        2. 65% Rule: lock profit at +2.5% (50%) and +5.0% (65%)
        3. Runner Hug: ATR trailing stop (3.0x → 0.8x → 0.6x as profit grows)
        """
        if len(df) < self.sma_long:
            return {
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "total_wins": 0,
                "total_losses": 0,
                "error": "insufficient data",
            }

        df = self._compute_signals(df)
        # Drop rows where ATR is NaN (first atr_period rows)
        df = df.dropna(subset=["sma_short", "sma_long", "atr"])

        trades: List[Trade] = []
        current_trade: Optional[Trade] = None
        # Daily strategy returns (0 when flat) for a Sharpe that is
        # comparable with the daily-return baselines.
        daily_returns = np.zeros(len(df))
        last_mark = None

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            open_at_bar_start = current_trade

            # --- Entry: SMA cross up ---
            if current_trade is None and prev_row["position"] == 1:
                current_trade = Trade(
                    entry_date=row.name,
                    entry_price=row["Close"],
                    entry_atr=row["atr"],
                    stop_price=row["Close"] - self.atr_mult_initial * row["atr"],
                    qty=1.0,
                )
                current_trade.exit_reason = ""
                last_mark = float(row["Close"])

            # --- Exit logic for open trade ---
            if current_trade is not None:
                current_trade.max_gain_pct = max(
                    current_trade.max_gain_pct,
                    (row["Close"] - current_trade.entry_price) / current_trade.entry_price * 100,
                )

                exit_price = None
                exit_reason = None

                # Check 1: SMA crossover (short crosses below long)
                if prev_row["position"] == -1:
                    exit_price = row["Close"]
                    exit_reason = "sma_crossover"

                # Check 2: 65% Rule — hard profit floor
                if exit_price is None:
                    gain_pct = (row["Close"] - current_trade.entry_price) / current_trade.entry_price
                    if gain_pct >= self.profit_floor_2:
                        # Lock in 65% of total gain
                        locked_gain = gain_pct * self.profit_lock_2
                        floor_price = current_trade.entry_price * (1 + locked_gain)
                        # Stop can only tighten (move up for long positions)
                        if exit_price is None or floor_price > current_trade.stop_price:
                            current_trade.stop_price = max(current_trade.stop_price, floor_price)
                            if row["Close"] < current_trade.stop_price:
                                exit_price = current_trade.stop_price
                                exit_reason = "65pct_rule_second"

                if exit_price is None:
                    # Check first threshold: +2.5% → lock 50%
                    gain_pct = (row["Close"] - current_trade.entry_price) / current_trade.entry_price
                    if gain_pct >= self.profit_floor_1:
                        locked_gain = gain_pct * self.profit_lock_1
                        floor_price = current_trade.entry_price * (1 + locked_gain)
                        current_trade.stop_price = max(current_trade.stop_price, floor_price)

                # Check 3: Runner Hug — ATR trailing stop
                if exit_price is None:
                    gain_pct = (row["Close"] - current_trade.entry_price) / current_trade.entry_price
                    if gain_pct >= self.profit_floor_2:
                        atr_mult = self.atr_mult_tightest  # 0.6x
                    elif gain_pct >= self.profit_floor_1:
                        atr_mult = self.atr_mult_tight  # 0.8x
                    else:
                        atr_mult = self.atr_mult_initial  # 3.0x

                    trailing_stop = row["Close"] - atr_mult * row["atr"]
                    current_trade.stop_price = max(current_trade.stop_price, trailing_stop)

                    # Exit if price closes below trailing stop
                    if row["Low"] < current_trade.stop_price:
                        exit_price = current_trade.stop_price
                        exit_reason = f"runner_hug_{atr_mult}x_atr"

                if open_at_bar_start is not None and last_mark:
                    mark = float(exit_price) if exit_price is not None else float(row["Close"])
                    daily_returns[i] = mark / last_mark - 1
                    last_mark = mark

                # Execute exit
                if exit_price is not None:
                    current_trade.exit_date = row.name
                    current_trade.exit_price = exit_price
                    current_trade.exit_reason = exit_reason
                    current_trade.status = "closed"
                    trades.append(current_trade)
                    current_trade = None

        # Close any remaining open position at last price
        if current_trade is not None:
            current_trade.exit_date = df.iloc[-1].name
            current_trade.exit_price = df.iloc[-1]["Close"]
            current_trade.exit_reason = "end_of_data"
            current_trade.status = "closed"
            trades.append(current_trade)

        # Compute performance metrics
        returns = [t.pnl_pct() for t in trades]
        if not returns:
            return {
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "total_wins": 0,
                "total_losses": 0,
            }

        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        # Return / drawdown / Sharpe from the DAILY mark-to-market curve so
        # they are directly comparable with the daily-return baselines.
        daily = pd.Series(daily_returns, index=df.index)
        dm = metrics_from_daily(daily.iloc[1:])
        total_return = dm["total_return_pct"]
        max_dd = dm["max_drawdown_pct"]
        sharpe = dm["sharpe_ratio"]

        return {
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "total_trades": len(trades),
            "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
            "avg_win_pct": round(float(np.mean(wins)), 2) if wins else 0,
            "avg_loss_pct": round(float(np.mean(losses)), 2) if losses else 0,
            "total_wins": len(wins),
            "total_losses": len(losses),
            "trade_returns": returns,
            "exit_reasons": {r: sum(1 for t in trades if t.exit_reason == r) for r in set(t.exit_reason for t in trades)},
            "daily_returns": daily,
            "trades": [
                {"entry_date": t.entry_date, "exit_date": t.exit_date,
                 "pnl_pct": t.pnl_pct(), "exit_reason": t.exit_reason}
                for t in trades
            ],
        }
