"""
Forecast recording — append-only, immutable records of every candidate decision.

Each forecast captures the decision context at a single point in time:
  - Decision time, symbol, horizon
  - Expected return, confidence, thesis
  - Source publication times, strategy version, data snapshot

Forecasts are NEVER revised after recording. If a correction is needed,
a new forecast is recorded with a reference to the original.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional
from enum import Enum


class ForecastStatus(str, Enum):
    PENDING = "pending"
    EVALUATED = "evaluated"
    EXPIRED = "expired"


FORECAST_HEADERS = [
    "forecast_id",
    "decision_time",
    "symbol",
    "horizon_days",
    "expected_return_pct",
    "confidence",
    "thesis",
    "source_publication_times",
    "strategy_version",
    "data_snapshot_id",
    "status",
    "created_at",
]


@dataclass
class Forecast:
    """
    An immutable record of a candidate decision.

    Captures the full decision context at a single point in time.
    Once recorded, a forecast is never modified.
    """
    forecast_id: str
    decision_time: str
    symbol: str
    horizon_days: int
    expected_return_pct: float
    confidence: float
    thesis: str
    source_publication_times: str  # JSON array of source timestamps
    strategy_version: str
    data_snapshot_id: str
    status: ForecastStatus = ForecastStatus.PENDING
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    @property
    def horizon_end(self) -> Optional[datetime]:
        """Compute the horizon end date from decision_time + horizon_days."""
        try:
            dt = datetime.fromisoformat(self.decision_time)
            # Add trading days (approximation: calendar days * 7/5)
            calendar_days = int(self.horizon_days * 7 / 5) + 1
            return dt + timedelta(days=calendar_days)
        except (ValueError, TypeError):
            return None

    def is_matured(self, now: Optional[datetime] = None) -> bool:
        """Check if the forecast horizon has elapsed."""
        end = self.horizon_end
        if end is None:
            return False
        if now is None:
            now = datetime.now(timezone.utc)
        # Make timezone-aware comparison
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        return now >= end

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Forecast":
        data = dict(data)
        status_str = data.pop("status", "pending")
        if isinstance(status_str, str):
            status = ForecastStatus(status_str)
        else:
            status = status_str
        return cls(status=status, **data)


def generate_forecast_id(symbol: str, decision_time: str, thesis: str) -> str:
    """Generate a unique forecast ID."""
    raw = f"{symbol}|{decision_time}|{thesis[:50]}"
    hash_suffix = hashlib.sha256(raw.encode()).hexdigest()[:8]
    # Clean timestamp for ID
    ts = decision_time.replace(":", "").replace("-", "").replace(".", "").replace("+", "")[:14]
    return f"fc-{symbol}-{ts}-{hash_suffix}"


class ForecastStore:
    """
    Persistent append-only storage for forecasts.

    Stored as CSV for auditability. Each forecast is a row with
    full decision context. The file is never rewritten — only appended.
    """

    def __init__(self, path: str = "learning/forecasts.csv"):
        self.path = path
        self._ensure_file()

    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(FORECAST_HEADERS)

    def record(self, forecast: Forecast) -> Forecast:
        """
        Record a new forecast. Append-only — never modifies existing rows.

        Returns the recorded forecast.
        """
        # Check for duplicate forecast_id
        existing = self.get(forecast.forecast_id)
        if existing is not None:
            raise ValueError(
                f"Forecast {forecast.forecast_id} already exists. "
                "Forecasts are immutable — record a new one instead."
            )

        row = [
            forecast.forecast_id,
            forecast.decision_time,
            forecast.symbol,
            forecast.horizon_days,
            f"{forecast.expected_return_pct:.4f}",
            f"{forecast.confidence:.4f}",
            forecast.thesis,
            forecast.source_publication_times,
            forecast.strategy_version,
            forecast.data_snapshot_id,
            forecast.status.value,
            forecast.created_at,
        ]
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        return forecast

    def get(self, forecast_id: str) -> Optional[Forecast]:
        """Retrieve a forecast by ID."""
        for f in self.read_all():
            if f.forecast_id == forecast_id:
                return f
        return None

    def read_all(self) -> list[Forecast]:
        """Read all forecasts."""
        if not os.path.exists(self.path):
            return []
        forecasts = []
        with open(self.path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    row["horizon_days"] = int(row["horizon_days"])
                    row["expected_return_pct"] = float(row["expected_return_pct"])
                    row["confidence"] = float(row["confidence"])
                    if "status" in row and isinstance(row["status"], str):
                        row["status"] = ForecastStatus(row["status"])
                    forecasts.append(Forecast(**row))
                except Exception:
                    continue
        return forecasts

    def get_by_status(self, status: ForecastStatus) -> list[Forecast]:
        """Get all forecasts with a given status."""
        return [f for f in self.read_all() if f.status == status]

    def get_by_symbol(self, symbol: str) -> list[Forecast]:
        """Get all forecasts for a symbol."""
        return [f for f in self.read_all() if f.symbol == symbol]

    def get_matured(self, now: Optional[datetime] = None) -> list[Forecast]:
        """Get all forecasts whose horizon has elapsed."""
        if now is None:
            now = datetime.now(timezone.utc)
        return [f for f in self.read_all() if f.is_matured(now)]

    def update_status(self, forecast_id: str, new_status: ForecastStatus) -> bool:
        """
        Update the status of a forecast. This is the ONLY field that can
        be modified after recording — all other fields are immutable.

        Status transitions: PENDING → EVALUATED, PENDING → EXPIRED
        """
        forecasts = self.read_all()
        found = False
        for f in forecasts:
            if f.forecast_id == forecast_id:
                f.status = new_status
                found = True
                break

        if not found:
            return False

        # Rewrite the file (only status field changes)
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FORECAST_HEADERS)
            for fc in forecasts:
                writer.writerow([
                    fc.forecast_id,
                    fc.decision_time,
                    fc.symbol,
                    fc.horizon_days,
                    f"{fc.expected_return_pct:.4f}",
                    f"{fc.confidence:.4f}",
                    fc.thesis,
                    fc.source_publication_times,
                    fc.strategy_version,
                    fc.data_snapshot_id,
                    fc.status.value,
                    fc.created_at,
                ])
        return True

    def get_stats(self) -> dict:
        """Get summary statistics."""
        forecasts = self.read_all()
        by_status = {}
        for f in forecasts:
            s = f.status.value
            by_status[s] = by_status.get(s, 0) + 1

        by_symbol = {}
        for f in forecasts:
            by_symbol[f.symbol] = by_symbol.get(f.symbol, 0) + 1

        return {
            "total": len(forecasts),
            "by_status": by_status,
            "by_symbol": by_symbol,
        }
