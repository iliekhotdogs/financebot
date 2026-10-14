"""
Strict response schemas for the Advisor and Jev models.

Both models return JSON that is validated against these schemas. If the
response is invalid, times out, or errors, the cycle fails safe — no
trade is opened and the reason is logged to the audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Literal
from datetime import datetime, timezone


# ------------------------------------------------------------------ #
# Advisor response schema
# ------------------------------------------------------------------ #

@dataclass
class AdvisorResponse:
    """
    Validated output from the advisor model.

    The advisor provides *advice only* — it has no authority to
    place orders, access exchange credentials, or bypass risk
    checks. Jev is the sole decision-maker.
    """
    recommendation: Literal["buy", "sell", "hold"]
    confidence: float  # 0.0 — 1.0
    evidence: str       # concise supporting evidence (max 500 chars)
    risks: str          # concise risk considerations (max 500 chars)
    schema_version: str = "1.0"
    raw: str = ""       # original text (for debugging, never logged with secrets)

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty = valid)."""
        errors = []
        if self.recommendation not in ("buy", "sell", "hold"):
            errors.append(f"recommendation '{self.recommendation}' not in buy/sell/hold")
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"confidence {self.confidence} outside [0, 1]")
        if not self.evidence:
            errors.append("evidence is empty")
        if not self.risks:
            errors.append("risks is empty")
        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "AdvisorResponse":
        """
        Build an AdvisorResponse from a dict (parsed JSON).
        Raises ValueError if required fields are missing or invalid.
        """
        if not isinstance(data, dict):
            raise ValueError("Advisor response must be a JSON object")

        rec = data.get("recommendation")
        if rec not in ("buy", "sell", "hold"):
            raise ValueError(
                f"Invalid recommendation '{rec}'; must be buy/sell/hold"
            )

        confidence = data.get("confidence")
        if confidence is None:
            raise ValueError("Missing 'confidence' field")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence {confidence} outside [0, 1]")

        # Never silently drop caveats or extra fields before Jev sees them.
        extra = set(data) - {"recommendation", "confidence", "evidence", "risks", "schema_version", "_raw"}
        if extra:
            raise ValueError(f"Unsupported advisor fields: {', '.join(sorted(extra))}")
        evidence = data.get("evidence")
        risks = data.get("risks")
        for name, value in (("evidence", evidence), ("risks", risks)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing or empty '{name}' field")
            if len(value) > 500:
                raise ValueError(f"Advisor '{name}' exceeds 500 characters; cannot forward intact")

        return cls(
            recommendation=rec,
            confidence=confidence,
            evidence=evidence,
            risks=risks,
            schema_version=str(data.get("schema_version", "1.0")),
            raw=data.get("_raw", ""),
        )


# ------------------------------------------------------------------ #
# Jev decision schema
# ------------------------------------------------------------------ #

@dataclass
class JevDecision:
    """
    Validated output from Jev — the sole decision-maker.

    Jev receives the snapshot, the strategy's valid actions, and the
    advisor's recommendation. Jev's decision is final.
    """
    action: Literal["buy", "sell", "hold"]
    qty: float
    stop_price: Optional[float]   # None if hold
    rationale: str                  # concise explanation (max 500 chars)
    confidence: float             # 0.0 — 1.0
    schema_version: str = "1.0"
    raw: str = ""

    def validate(self) -> list[str]:
        errors = []
        if self.action not in ("buy", "sell", "hold"):
            errors.append(f"action '{self.action}' not in buy/sell/hold")
        if self.action != "hold":
            if self.qty <= 0:
                errors.append(f"qty {self.qty} must be positive for non-hold actions")
            if self.stop_price is None:
                errors.append("stop_price is required for buy/sell actions")
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"confidence {self.confidence} outside [0, 1]")
        if not self.rationale:
            errors.append("rationale is empty")
        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "JevDecision":
        if not isinstance(data, dict):
            raise ValueError("Jev response must be a JSON object")

        action = data.get("action")
        if action not in ("buy", "sell", "hold"):
            raise ValueError(
                f"Invalid action '{action}'; must be buy/sell/hold"
            )

        qty = data.get("qty", 0)
        qty = float(qty) if qty is not None else 0.0

        stop_price = data.get("stop_price")
        if stop_price is not None:
            stop_price = float(stop_price)

        rationale = data.get("rationale", "")
        if not rationale:
            raise ValueError("Missing or empty 'rationale' field")

        confidence = data.get("confidence")
        if confidence is None:
            raise ValueError("Missing 'confidence' field")
        confidence = float(confidence)
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence {confidence} outside [0, 1]")

        return cls(
            action=action,
            qty=qty,
            stop_price=stop_price,
            rationale=rationale[:500],
            confidence=confidence,
            schema_version=str(data.get("schema_version", "1.0")),
            raw=data.get("_raw", ""),
        )


# ------------------------------------------------------------------ #
# Snapshot
# ------------------------------------------------------------------ #

@dataclass
class Snapshot:
    """
    A single timestamped point-in-time market/account/strategy snapshot.

    This same snapshot is sent to both the advisor and Jev so that
    neither sees future prices or post-trade outcomes.
    """
    snapshot_id: str
    timestamp: str            # ISO 8601 UTC
    symbol: str
    market: dict              # latest bar, close, SMA values, etc.
    account: dict             # cash, portfolio value, currency
    position: Optional[dict]  # current position for this symbol (if any)
    strategy_signal: int      # -1, 0, 1
    strategy_name: str
    valid_actions: list       # ["buy", "sell", "hold"] — depends on position state

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "market": self.market,
            "account": self.account,
            "position": self.position,
            "strategy_signal": self.strategy_signal,
            "strategy_name": self.strategy_name,
            "valid_actions": self.valid_actions,
        }

    def summary_text(self) -> str:
        """Human-readable summary for sending to the models."""
        pos = self.position or {}
        lines = [
            f"Timestamp (UTC): {self.timestamp}",
            f"Symbol: {self.symbol}",
            f"Strategy: {self.strategy_name}",
            f"Signal: {self.strategy_signal} (1=buy, -1=sell, 0=hold)",
            f"Valid actions: {self.valid_actions}",
            f"Latest close: {self.market.get('close', 'N/A')}",
            f"SMA short: {self.market.get('sma_short', 'N/A')}",
            f"SMA long: {self.market.get('sma_long', 'N/A')}",
            f"Cash: {self.account.get('cash', 'N/A')}",
            f"Portfolio value: {self.account.get('portfolio_value', 'N/A')}",
            f"Current position: {pos.get('symbol', 'none')} "
            f"qty={pos.get('qty', 'N/A')} avg_entry={pos.get('avg_entry_price', 'N/A')}",
        ]
        return "\n".join(lines)


# ------------------------------------------------------------------ #
# Fail-safe result wrapper
# ------------------------------------------------------------------ #

@dataclass
class ModelResult:
    """
    Wrapped result from an LLM call. On any error, timeout, or
    schema validation failure, success=False and the cycle fails safe.
    """
    success: bool
    data: Optional[object] = None
    error: Optional[str] = None
    latency_ms: int = 0
    snapshot_ref: str = ""
