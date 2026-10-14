"""
Trade log: append-only CSV.

Schema:
  timestamp, symbol, action, qty, price, order_id, reason, source, status

- Timestamps in UTC ISO 8601
- action: buy | sell | hold
- source: alpaca | yfinance-sim
- status: filled | pending | rejected | simulated
"""

import csv
import os
from datetime import datetime, timezone

LOG_PATH = os.path.join(os.path.dirname(__file__), "trade_log.csv")

HEADERS = [
    "timestamp", "symbol", "action", "qty", "price",
    "order_id", "reason", "source", "status",
]


def ensure_header(path=None):
    path = path or LOG_PATH
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)


def log_trade(symbol, action, qty, price, order_id="", reason="",
              source="alpaca", status="filled", path=None):
    path = path or LOG_PATH
    ensure_header(path)
    row = [
        datetime.now(timezone.utc).isoformat(),
        symbol, action, qty, price,
        order_id, reason, source, status,
    ]
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    return row
