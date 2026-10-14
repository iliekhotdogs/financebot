"""
Learning audit log — records every candidate, rejection, promotion,
rollback, and operator approval.

This is a separate audit trail from the main trading audit log.
It tracks the learning system's decisions and is fully reproducible.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Optional


LEARNING_AUDIT_HEADERS = [
    "timestamp",
    "event_type",       # candidate_created | evaluation_complete | promotion_assessed |
                        # promoted | rejected | rolled_back | operator_approval |
                        # operator_rejection | config_changed
    "candidate_id",
    "candidate_version",
    "details",          # JSON blob with event-specific details
    "operator_id",      # who approved/rejected (if applicable)
    "previous_state",   # state before the event
    "new_state",        # state after the event
    "rationale",        # why this event occurred
]


class LearningAuditLog:
    """
    Persistent audit log for the learning system.

    Every candidate lifecycle event is recorded with full context.
    This log is append-only and never modified.
    """

    def __init__(self, path: str = "learning/learning_audit_log.csv"):
        self.path = path
        self._ensure_file()

    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(LEARNING_AUDIT_HEADERS)

    def log_event(
        self,
        event_type: str,
        candidate_id: str = "",
        candidate_version: str = "",
        details: dict = None,
        operator_id: str = "",
        previous_state: str = "",
        new_state: str = "",
        rationale: str = "",
    ):
        """Record a learning system event."""
        import json
        row = [
            datetime.now(timezone.utc).isoformat(),
            event_type,
            candidate_id,
            candidate_version,
            json.dumps(details or {}),
            operator_id,
            previous_state,
            new_state,
            rationale,
        ]
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        return row

    def read_all(self) -> list[dict]:
        """Read all audit events."""
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            reader = csv.DictReader(f)
            return list(reader)

    def get_events_for_candidate(self, candidate_id: str) -> list[dict]:
        """Get all events for a specific candidate."""
        return [e for e in self.read_all() if e.get("candidate_id") == candidate_id]

    def get_events_by_type(self, event_type: str) -> list[dict]:
        """Get all events of a specific type."""
        return [e for e in self.read_all() if e.get("event_type") == event_type]
