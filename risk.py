"""
Risk management — unified risk limits for the combined trading bot.

Paper trading mode:  yfinance (backfill) + Alpaca (real-time + paper execution).
Live trading mode:    Liquid Trade (perps + execution) — enabled only after
                      paper trading is validated and operator approves.

Risk limits are documented in config.yaml and enforced before every order.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RiskCheck:
    """Result of a risk limit check."""
    passed: bool
    reason: str
    action: str  # 'allow' | 'reject' | 'halt'


@dataclass
class DailyPnL:
    """Tracks daily profit/loss across closed trades."""
    date: str  # ISO date string
    realized_pl: float = 0.0
    trade_count: int = 0
    max_risk_used: float = 0.0

    def loss(self) -> float:
        """Returns positive value for a loss, 0 for break-even/profit."""
        return max(0.0, -self.realized_pl)


class RiskManager:
    """
    Enforces pre-trade risk limits and daily loss tracking.

    Limits (all in config.yaml → risk_limits):
      - per_trade_max_loss_pct / _usd  : hard stop per position
      - total_open_exposure_pct / _usd : caps concentration
      - max_position_size_pct          : cap on a single position notional
      - daily_loss_limit_pct / _usd    : hard stop for the day
      - drawdown_halt_pct / _usd       : peak-to-trough halt
      - max_leverage                   : start at 1.0 (no leverage)
      - min_order_collateral_usd       : platform minimum
    """

    def __init__(self, config: dict, initial_balance: float = 100000.0):
        self.limits = config.get("risk_limits", {})
        self.balance = float(initial_balance)
        self.peak_balance = self.balance
        self.halted = False
        self.halt_reason: Optional[str] = None
        # daily P&L tracking: {date_str: DailyPnL}
        self._daily: dict[str, DailyPnL] = defaultdict(
            lambda: DailyPnL(date=date.today().isoformat())
        )
        # current open exposure: {symbol: notional}
        self._open_exposure: dict[str, float] = defaultdict(float)

    # ------------------------------------------------------------------ #
    # Daily tracking
    # ------------------------------------------------------------------ #
    @property
    def today(self) -> str:
        return date.today().isoformat()

    def _today_pnl(self) -> DailyPnL:
        return self._daily[self.today]

    def record_trade_result(self, symbol: str, realized_pl: float,
                            qty: float, price: float) -> None:
        """Record a closed-trade P&L for daily loss-limit tracking."""
        dp = self._today_pnl()
        dp.realized_pl += realized_pl
        dp.trade_count += 1
        # reduce exposure
        notional = abs(qty) * price
        self._open_exposure[symbol] = max(0.0, self._open_exposure[symbol] - notional)

    def release_position(self, symbol: str) -> None:
        """Drop a closed position's exposure (day P&L comes from broker equity)."""
        self._open_exposure.pop(symbol, None)

    def record_position(self, symbol: str, notional: float) -> None:
        """Add to open exposure for a symbol."""
        self._open_exposure[symbol] += notional

    def update_balance(self, new_balance: float) -> None:
        """Update account balance (from broker), track peak for drawdown."""
        self.balance = float(new_balance)
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

    # ------------------------------------------------------------------ #
    # Broker sync + persistence
    #
    # Each cron cycle is a fresh process, so in-memory state never
    # survives. Rebuild exposure/balance/day P&L from the broker at cycle
    # start and persist only what the broker cannot tell us (peak equity,
    # halt flag).
    # ------------------------------------------------------------------ #
    def sync_from_broker(self, account: dict, positions: list) -> None:
        """Rebuild balance, open exposure and today's P&L from broker state."""
        equity = account.get("equity") or account.get("portfolio_value")
        if equity is not None:
            self.update_balance(float(equity))
        last_equity = account.get("last_equity")
        if equity is not None and last_equity not in (None, ""):
            # Equity change since previous close (realized + unrealized).
            # Conservative for a daily loss limit.
            self._today_pnl().realized_pl = float(equity) - float(last_equity)
        self._open_exposure = defaultdict(float)
        for p in positions:
            mv = p.get("market_value")
            if mv in (None, ""):
                mv = float(p.get("qty", 0)) * float(p.get("current_price") or p.get("avg_entry_price") or 0)
            self._open_exposure[p.get("symbol")] = abs(float(mv))

    def load_state(self, path: str) -> None:
        """Restore peak balance and halt state persisted by a previous cycle."""
        try:
            with open(path) as f:
                st = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self.peak_balance = max(self.peak_balance, float(st.get("peak_balance", 0) or 0))
        if st.get("halted"):
            self.halted = True
            self.halt_reason = st.get("halt_reason") or "Halted in a previous cycle"

    def save_state(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "peak_balance": self.peak_balance,
                "balance": self.balance,
                "halted": self.halted,
                "halt_reason": self.halt_reason,
                "updated_at": datetime.now().isoformat(),
            }, f, indent=2)
        os.replace(tmp, path)

    def size_position(self, entry_price: float, stop_price: Optional[float]) -> int:
        """
        Whole-share size satisfying every entry limit:
          min(max position notional, per-trade risk budget / stop distance,
              remaining exposure capacity) / price
        Never sizes up to recover a loss.
        """
        if entry_price <= 0:
            return 0
        max_pos_usd = self.cap("max_position_size", 5.0)
        notional_cap = min(max_pos_usd, self.available_exposure_capacity())
        qty = notional_cap / entry_price
        if stop_price is not None and 0 < stop_price < entry_price:
            max_loss_usd = self.cap("per_trade_max_loss", 2.0)
            qty = min(qty, max_loss_usd / (entry_price - stop_price))
        return max(0, int(qty))

    # ------------------------------------------------------------------ #
    # Limit checks
    # ------------------------------------------------------------------ #
    def cap(self, name: str, default_pct: float) -> float:
        """
        Effective $ limit = min(<name>_pct of CURRENT equity, <name>_usd).
        Previously the fixed _usd value always won, so limits never shrank
        after losses (or grew with the account).
        """
        pct = self.limits.get(f"{name}_pct", default_pct)
        by_pct = self.balance * float(pct) / 100
        usd = self.limits.get(f"{name}_usd")
        return min(by_pct, float(usd)) if usd is not None else by_pct

    def total_open_exposure(self) -> float:
        return sum(self._open_exposure.values())

    def available_exposure_capacity(self) -> float:
        return max(0.0, self.cap("total_open_exposure", 10.0) - self.total_open_exposure())

    def check_trade(self, symbol: str, side: str, qty: float,
                    entry_price: float, stop_price: Optional[float],
                    held_qty: float = 0.0) -> RiskCheck:
        """
        Pre-trade risk gate. All limits must pass.

        side: 'buy' or 'sell'
        qty: number of shares
        entry_price: intended fill price
        stop_price: stop-loss price (for risk calc)
        held_qty: shares currently held long in `symbol`

        A sell that closes (part of) an existing long REDUCES risk, so it is
        always allowed — including while halted (shutdown policy: close what
        can be closed). Entry limits apply only to risk-increasing orders.
        """
        if side == "sell" and held_qty > 0:
            if qty <= 0 or qty > held_qty + 1e-9:
                return RiskCheck(
                    False, f"Sell qty {qty} invalid for held qty {held_qty}", "reject",
                )
            return RiskCheck(True, "Closing/reducing existing position (always allowed)", "allow")
        if side == "sell":
            # Long-only bot: a sell without a position would open a short.
            return RiskCheck(False, "Short selling is not allowed (no position to close)", "reject")

        if self.halted:
            return RiskCheck(False, self.halt_reason or "Trading halted", "halt")

        # --- drawdown halt (checked first: it must bind even if the order
        #     would fail other limits) ---
        dd_pct = self.limits.get("drawdown_halt_pct", 15.0)
        dd_threshold = self.peak_balance * (1 - dd_pct / 100)
        if self.balance < dd_threshold:
            self.halted = True
            self.halt_reason = (
                f"Drawdown halt: balance ${self.balance:.2f} < "
                f"peak ${self.peak_balance:.2f} threshold ${dd_threshold:.2f} ({dd_pct}%)"
            )
            return RiskCheck(False, self.halt_reason, "halt")

        if stop_price is None:
            stop_price = entry_price
        notional = abs(qty) * entry_price

        # --- min order size ---
        min_col = self.limits.get("min_order_collateral_usd", 15.0)
        if notional < min_col:
            return RiskCheck(
                False,
                f"Order notional ${notional:.2f} < min ${min_col:.2f}",
                "reject",
            )

        # --- per-trade max loss ---
        risk_per_share = abs(entry_price - stop_price)
        risk_amount = qty * risk_per_share
        max_loss_usd = self.cap("per_trade_max_loss", 2.0)
        if risk_amount > max_loss_usd:
            return RiskCheck(
                False,
                f"Per-trade risk ${risk_amount:.2f} > max ${max_loss_usd:.2f}",
                "reject",
            )

        # --- max position size ---
        max_pos_usd = self.cap("max_position_size", 5.0)
        if notional > max_pos_usd:
            return RiskCheck(
                False,
                f"Position size ${notional:.2f} > max ${max_pos_usd:.2f}",
                "reject",
            )

        # --- total open exposure ---
        exp_limit_usd = self.cap("total_open_exposure", 10.0)
        new_total = self.total_open_exposure() + notional
        if new_total > exp_limit_usd:
            remaining = exp_limit_usd - self.total_open_exposure()
            return RiskCheck(
                False,
                f"Total exposure ${new_total:.2f} > limit ${exp_limit_usd:.2f}; "
                f"only ${remaining:.2f} remaining",
                "reject",
            )

        # --- daily loss limit ---
        dp = self._today_pnl()
        daily_limit_usd = self.cap("daily_loss_limit", 5.0)
        if dp.loss() >= daily_limit_usd:
            self.halted = True
            self.halt_reason = (
                f"Daily loss ${dp.loss():.2f} reached limit ${daily_limit_usd:.2f}"
            )
            return RiskCheck(False, self.halt_reason, "halt")
        # Projected loss if this trade hits stop
        projected_daily_loss = dp.loss() + risk_amount
        if projected_daily_loss > daily_limit_usd:
            return RiskCheck(
                False,
                f"Projected daily loss ${projected_daily_loss:.2f} > "
                f"limit ${daily_limit_usd:.2f}",
                "halt",
            )

        return RiskCheck(True, "All risk checks passed", "allow")

    def check_max_leverage(self, requested_leverage: float) -> RiskCheck:
        """Check if requested leverage is within limits."""
        max_lev = self.limits.get("max_leverage", 1.0)
        if requested_leverage > max_lev:
            return RiskCheck(
                False,
                f"Requested leverage {requested_leverage}x > max {max_lev}x",
                "reject",
            )
        return RiskCheck(True, f"Leverage {requested_leverage}x OK", "allow")

    def daily_summary(self) -> dict:
        """Return today's P&L summary."""
        dp = self._today_pnl()
        return {
            "date": dp.date,
            "realized_pl": dp.realized_pl,
            "trade_count": dp.trade_count,
            "loss": dp.loss(),
            "daily_loss_limit_usd": self.cap("daily_loss_limit", 5.0),
            "halted": self.halted,
            "current_balance": self.balance,
            "peak_balance": self.peak_balance,
            "total_exposure": self.total_open_exposure(),
        }

    def reset_daily(self) -> None:
        """Called at EOD — resets daily P&L tracking."""
        # Keep history but start fresh daily P&L
        today = self.today
        old = self._daily.get(today)
        if old:
            logger.info(
                "EOD reset: %s realized_pl=%.2f trades=%d",
                today, old.realized_pl, old.trade_count,
            )
        # Don't clear history — keep for audit
        # But reset open exposure if all positions closed
