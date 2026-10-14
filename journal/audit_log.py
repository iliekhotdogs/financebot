"""
Extended audit trail for the advisory decision flow.

The original trade_log.csv schema is preserved unchanged. This module
adds a parallel audit_log.csv that captures the full decision chain:
  snapshot reference → advisor output → Jev decision → risk check → execution.

This provides the data the dashboard and daily report need to show
advisor advice side-by-side with Jev's final decision.
"""
import csv
import os
from datetime import datetime, timezone
from typing import Optional

LOG_PATH = os.path.join(os.path.dirname(__file__), "audit_log.csv")

# Extended audit schema — captures full decision chain per symbol per cycle
AUDIT_HEADERS = [
    "timestamp",          # UTC ISO 8601
    "snapshot_id",        # reference to the snapshot dict
    "symbol",
    "strategy_name",
    "strategy_signal",    # -1, 0, 1
    "advisor_enabled",    # bool
    "advisor_recommendation",  # buy/sell/hold
    "advisor_confidence",      # 0.0 — 1.0
    "advisor_evidence",        # trimmed to 500 chars
    "advisor_risks",            # trimmed to 500 chars
    "advisor_error",            # if advisor failed
    "jev_action",              # buy/sell/hold (final decision)
    "jev_qty",                 # proposed quantity
    "jev_stop_price",          # proposed stop
    "jev_rationale",           # trimmed to 500 chars
    "jev_confidence",          # 0.0 — 1.0
    "jev_error",               # if Jev failed
    "risk_passed",             # bool
    "risk_action",             # allow/reject/halt
    "risk_reason",             # trimmed to 500 chars
    "execution_status",        # filled/rejected/failed/not_attempted/skipped
    "order_id",                # Alpaca order id if executed
    "execution_price",         # actual fill price
    "execution_error",         # if execution failed
    "cycle_outcome",           # completed/trade_opened/no_signal/advisor_failed/jev_failed/risk_rejected/execution_failed/manual_review
]


def ensure_audit_header(path: Optional[str] = None) -> None:
    """Create the audit log file with headers if it doesn't exist."""
    path = path or LOG_PATH
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(AUDIT_HEADERS)


def log_audit_event(
    snapshot_id: str,
    symbol: str,
    strategy_name: str,
    strategy_signal: int,
    advisor_enabled: bool,
    advisor_recommendation: Optional[str] = None,
    advisor_confidence: Optional[float] = None,
    advisor_evidence: str = "",
    advisor_risks: str = "",
    advisor_error: str = "",
    jev_action: Optional[str] = None,
    jev_qty: float = 0,
    jev_stop_price: Optional[float] = None,
    jev_rationale: str = "",
    jev_confidence: Optional[float] = None,
    jev_error: str = "",
    risk_passed: Optional[bool] = None,
    risk_action: str = "",
    risk_reason: str = "",
    execution_status: str = "not_attempted",
    order_id: str = "",
    execution_price: Optional[float] = None,
    execution_error: str = "",
    cycle_outcome: str = "",
    path: Optional[str] = None,
) -> list:
    """
    Append a full decision-chain audit record.

    Returns the row list written.
    """
    path = path or LOG_PATH
    ensure_audit_header(path)

    # Trim long text fields for CSV storage
    def _trim(s: str, maxlen: int = 500) -> str:
        if not s:
            return ""
        return s[:maxlen]

    row = [
        datetime.now(timezone.utc).isoformat(),
        snapshot_id,
        symbol,
        strategy_name,
        strategy_signal,
        str(advisor_enabled).lower(),
        advisor_recommendation or "",
        f"{advisor_confidence:.4f}" if advisor_confidence is not None else "",
        _trim(advisor_evidence),
        _trim(advisor_risks),
        _trim(advisor_error, maxlen=300),
        jev_action or "",
        f"{jev_qty}",
        f"{jev_stop_price:.2f}" if jev_stop_price is not None else "",
        _trim(jev_rationale),
        f"{jev_confidence:.4f}" if jev_confidence is not None else "",
        _trim(jev_error, maxlen=300),
        str(risk_passed).lower() if risk_passed is not None else "",
        risk_action,
        _trim(risk_reason),
        execution_status,
        order_id,
        f"{execution_price:.2f}" if execution_price is not None else "",
        _trim(execution_error, maxlen=300),
        cycle_outcome,
    ]

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return row


def read_audit_log(path: Optional[str] = None) -> list[dict]:
    """Read the audit log as a list of dicts (for dashboard/report)."""
    path = path or LOG_PATH
    if not os.path.exists(path):
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)
