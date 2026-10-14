"""
Candidate strategy definitions and storage.

Each candidate has:
  - A version identifier
  - A written hypothesis
  - Intended market conditions
  - Required data
  - A clear abstention rule
  - Parameter bounds (enforced)
  - Code/config hash for reproducibility

Candidates are proposed by the research process but NEVER modify
the active strategy. They exist only as evaluated alternatives.
"""

from __future__ import annotations

import hashlib
import json
import os
import csv
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional
from enum import Enum


class CandidateStatus(str, Enum):
    PROPOSED = "proposed"
    EVALUATING = "evaluating"
    PAPER = "paper_trading"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


@dataclass
class CandidateConfig:
    """Configuration bounds for a candidate strategy."""
    sma_short: int = 20
    sma_long: int = 50
    atr_period: int = 14
    atr_mult_initial: float = 3.0
    atr_mult_tight: float = 0.8
    atr_mult_tightest: float = 0.6
    profit_floor_1: float = 0.025
    profit_lock_1: float = 0.50
    profit_floor_2: float = 0.05
    profit_lock_2: float = 0.65
    # Parameter bounds — candidates cannot exceed these
    max_sma_short: int = 50
    min_sma_short: int = 5
    max_sma_long: int = 200
    min_sma_long: int = 20
    max_atr_mult: float = 5.0
    min_atr_mult: float = 0.5
    # Strategy family + full genome (strategies/families.py). Legacy rows
    # without these are SMA-cross candidates built from the fields above.
    family: str = "sma_cross"
    genes: dict = field(default_factory=dict)

    def validate_bounds(self) -> list[str]:
        """Check that parameters are within configured bounds."""
        errors = []
        if self.genes:
            from strategies.families import validate
            return validate(self.genes)
        if not (self.min_sma_short <= self.sma_short <= self.max_sma_short):
            errors.append(f"sma_short {self.sma_short} outside [{self.min_sma_short}, {self.max_sma_short}]")
        if not (self.min_sma_long <= self.sma_long <= self.max_sma_long):
            errors.append(f"sma_long {self.sma_long} outside [{self.min_sma_long}, {self.max_sma_long}]")
        if self.sma_short >= self.sma_long:
            errors.append(f"sma_short ({self.sma_short}) must be < sma_long ({self.sma_long})")
        if not (self.min_atr_mult <= self.atr_mult_initial <= self.max_atr_mult):
            errors.append(f"atr_mult_initial {self.atr_mult_initial} outside [{self.min_atr_mult}, {self.max_atr_mult}]")
        return errors


@dataclass
class Candidate:
    """
    A candidate strategy for evaluation.

    Candidates are proposed by the research process. They carry a
    written hypothesis, intended market conditions, required data,
    and a clear abstention rule. They NEVER modify the active strategy.
    """
    candidate_id: str
    version: str
    hypothesis: str
    intended_conditions: str
    required_data: str
    abstain_rule: str
    config: CandidateConfig
    status: CandidateStatus = CandidateStatus.PROPOSED
    created_at: str = ""
    code_hash: str = ""
    model_version: str = ""
    prompt_version: str = ""
    data_snapshot_id: str = ""
    parent_candidate_id: Optional[str] = None  # for tracking lineage
    notes: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.code_hash:
            self.code_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute a hash of the candidate config for reproducibility."""
        config_dict = asdict(self.config)
        raw = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["config"] = asdict(self.config)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Candidate":
        data = dict(data)  # don't mutate caller's dict
        config_data = data.pop("config", {})
        config = CandidateConfig(**config_data)
        status_str = data.pop("status", "proposed")
        if isinstance(status_str, str):
            status = CandidateStatus(status_str)
        else:
            status = status_str
        return cls(config=config, status=status, **data)


class CandidateStore:
    """
    Persistent storage for candidates.

    Stored as CSV for auditability. Each candidate is a row with
    full configuration, hypothesis, and status.
    """

    def __init__(self, path: str = "learning/candidates.csv"):
        self.path = path
        self._ensure_file()

    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "candidate_id", "version", "hypothesis", "intended_conditions",
                    "required_data", "abstain_rule", "status", "created_at",
                    "code_hash", "model_version", "prompt_version",
                    "data_snapshot_id", "parent_candidate_id", "notes",
                    "config_json",
                ])

    def save(self, candidate: Candidate):
        """Save or update a candidate."""
        candidates = self.read_all()
        # Remove existing entry with same ID
        candidates = [c for c in candidates if c.candidate_id != candidate.candidate_id]
        candidates.append(candidate)
        self._write_all(candidates)

    def get(self, candidate_id: str) -> Optional[Candidate]:
        for c in self.read_all():
            if c.candidate_id == candidate_id:
                return c
        return None

    def read_all(self) -> list[Candidate]:
        if not os.path.exists(self.path):
            return []
        candidates = []
        with open(self.path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    config = CandidateConfig(**json.loads(row.pop("config_json", "{}")))
                    row["config"] = config
                    # Convert status string to enum
                    if "status" in row and isinstance(row["status"], str):
                        row["status"] = CandidateStatus(row["status"])
                    candidates.append(Candidate(**row))
                except Exception:
                    continue
        return candidates

    def _write_all(self, candidates: list[Candidate]):
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "candidate_id", "version", "hypothesis", "intended_conditions",
                "required_data", "abstain_rule", "status", "created_at",
                "code_hash", "model_version", "prompt_version",
                "data_snapshot_id", "parent_candidate_id", "notes",
                "config_json",
            ])
            for c in candidates:
                writer.writerow([
                    c.candidate_id, c.version, c.hypothesis, c.intended_conditions,
                    c.required_data, c.abstain_rule, c.status.value, c.created_at,
                    c.code_hash, c.model_version, c.prompt_version,
                    c.data_snapshot_id, c.parent_candidate_id or "", c.notes,
                    json.dumps(asdict(c.config)),
                ])

    def get_by_status(self, status: CandidateStatus) -> list[Candidate]:
        return [c for c in self.read_all() if c.status == status]
