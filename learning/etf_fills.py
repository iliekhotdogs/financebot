"""Read-only reconciliation of ETF order fills against the paper broker."""
from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent.parent
TRADE_PATH = str(BOT_DIR / "journal" / "trade_log.csv")
FILL_PATH = str(BOT_DIR / "journal" / "etf_fills.csv")
FIELDS = ("order_id", "symbol", "side", "qty", "filled_qty", "planned_price",
          "filled_avg_price", "submitted_at", "filled_at", "status", "slippage_bp")
logger = logging.getLogger("finance-bot.etf-fills")


def _value(order, field):
    value = order.get(field) if isinstance(order, dict) else getattr(order, field, None)
    return getattr(value, "value", value)


def reconcile_fills(dp) -> list[dict]:
    """Append only newly filled ETF orders; brokerage operations are reads only."""
    try:
        if not os.path.exists(TRADE_PATH):
            return []
        with open(TRADE_PATH, newline="") as f:
            trades = list(csv.DictReader(f))
        if os.path.exists(FILL_PATH):
            with open(FILL_PATH, newline="") as f:
                existing = list(csv.DictReader(f))
        else:
            existing = []
        seen = {r["order_id"] for r in existing}
        new = []
        for trade in trades:
            oid = trade.get("order_id")
            if trade.get("source") != "etf_trend" or not oid or oid in seen:
                continue
            try:
                getter = getattr(dp, "get_order_by_id", None)
                order = getter(oid) if getter else dp.trading_client.get_order_by_id(oid)
                status = str(_value(order, "status")).lower()
                if status != "filled":
                    continue
                planned = float(trade["price"])
                actual = float(_value(order, "filled_avg_price"))
                side = trade["action"].lower()
                slippage = (actual - planned) / planned * 10000 * (1 if side == "buy" else -1)
                row = {"order_id": oid, "symbol": trade["symbol"], "side": side,
                       "qty": trade["qty"], "filled_qty": str(_value(order, "filled_qty")),
                       "planned_price": trade["price"], "filled_avg_price": str(actual),
                       "submitted_at": str(_value(order, "submitted_at") or ""),
                       "filled_at": str(_value(order, "filled_at") or ""),
                       "status": status, "slippage_bp": str(slippage)}
                new.append(row)
                seen.add(oid)
            except Exception as exc:
                logger.warning("Fill reconciliation failed for %s (%s); retry next run", oid, type(exc).__name__)
        if new:
            os.makedirs(os.path.dirname(FILL_PATH), exist_ok=True)
            with open(FILL_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                if not existing and f.tell() == 0:
                    writer.writeheader()
                writer.writerows(new)
        return existing + new
    except Exception as exc:
        logger.warning("Fill reconciliation failed (%s); retry next run", type(exc).__name__)
        return []
