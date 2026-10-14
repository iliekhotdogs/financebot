"""
Advisor model client — provides *advice only*.

The advisor receives the same snapshot that Jev receives. It returns a
recommendation with supporting evidence and risks, but it has NO access
to:
  - Order placement tools
  - Exchange credentials
  - The ability to bypass risk checks
  - Any post-trade outcome data

The advisor's output is advisory only; Jev makes the final decision.
If the advisor fails, times out, or returns malformed output, the
cycle fails safe (no trade opened) and the failure is logged.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

from schemas import AdvisorResponse, ModelResult
from snapshot import snapshot_to_text

logger = logging.getLogger(__name__)

# Default system prompt for the advisor
ADVISOR_SYSTEM_PROMPT = """You are an algorithmic trading advisor. Your job is to analyze a
point-in-time market snapshot and provide a concise recommendation
with supporting evidence and risk considerations.

You do NOT place trades. You do NOT access broker accounts. Your output
is advice only — Jev makes the final decision.

Respond with ONLY valid JSON matching this schema:
{
  "recommendation": "buy" | "sell" | "hold",
  "confidence": <float 0.0-1.0>,
  "evidence": "<concise evidence, max 500 chars>",
  "risks": "<concise risk considerations, max 500 chars>"
}

No extra text, no markdown, no explanations outside the JSON.
"""


class AdvisorClient:
    """
    Client for the advisor model.

    Configurable provider, model, timeout, and enable/disable.
    Credentials are read from environment variables — never logged.

    Supported providers:
      - "openai": OpenAI API-compatible (OpenAI, OpenRouter, etc.)
      - "anthropic": Anthropic Claude API
      - "mock": Returns a deterministic mock response (for testing)
    """

    def __init__(self, config: dict | None = None):
        adv_cfg = (config or {}).get("advisor", {})
        self.enabled = adv_cfg.get("enabled", False)
        self.provider = adv_cfg.get("provider", "openai")
        self.model = adv_cfg.get("model", "gpt-4o-mini")
        self.timeout = adv_cfg.get("timeout_seconds", 30)
        self.api_key = adv_cfg.get("api_key_env", "ADVISOR_API_KEY")
        self.base_url = adv_cfg.get("base_url", None)  # e.g. "http://localhost:20128/v1"
        # Don't store the actual key — read from env at call time
        self._api_key: Optional[str] = None

    def _get_api_key(self) -> Optional[str]:
        """Read API key from environment at call time (never logged)."""
        import os
        if self._api_key is None:
            env_name = self.api_key
            self._api_key = os.getenv(env_name)
        return self._api_key

    def analyze(self, snapshot: dict) -> ModelResult:
        """
        Send the snapshot to the advisor model and return a validated
        ModelResult.

        On any failure (disabled, timeout, malformed JSON, schema
        validation error), returns success=False with the error message.
        """
        if not self.enabled:
            return ModelResult(
                success=False,
                error="Advisor is disabled in configuration",
                snapshot_ref=snapshot.get("snapshot_id", ""),
            )

        snapshot_ref = snapshot.get("snapshot_id", "")
        prompt = snapshot_to_text(snapshot)

        start = time.monotonic()

        try:
            if self.provider == "mock":
                return self._analyze_mock(snapshot, prompt, start, snapshot_ref)
            elif self.provider == "openai":
                return self._analyze_openai(prompt, start, snapshot_ref)
            elif self.provider == "anthropic":
                return self._analyze_anthropic(prompt, start, snapshot_ref)
            else:
                return ModelResult(
                    success=False,
                    error=f"Unknown advisor provider: {self.provider}",
                    latency_ms=int((time.monotonic() - start) * 1000),
                    snapshot_ref=snapshot_ref,
                )
        except Exception as e:
            logger.error("Advisor call failed: %s", e)
            return ModelResult(
                success=False,
                error=f"Advisor exception: {type(e).__name__}: {e}",
                latency_ms=int((time.monotonic() - start) * 1000),
                snapshot_ref=snapshot_ref,
            )

    def _analyze_openai(self, prompt: str, start: float, snapshot_ref: str) -> ModelResult:
        """Call OpenAI-compatible API."""
        try:
            from openai import OpenAI
        except ImportError:
            return ModelResult(
                success=False,
                error="openai package not installed",
                latency_ms=int((time.monotonic() - start) * 1000),
                snapshot_ref=snapshot_ref,
            )

        api_key = self._get_api_key()
        if not api_key:
            return ModelResult(
                success=False,
                error="ADVISOR_API_KEY not set in environment",
                latency_ms=int((time.monotonic() - start) * 1000),
                snapshot_ref=snapshot_ref,
            )

        client = OpenAI(api_key=api_key, timeout=self.timeout, base_url=self.base_url)

        messages = [
            {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        last_error = "No response"
        import openai as _openai
        transient = tuple(c for c in (json.JSONDecodeError,
                                      getattr(_openai, "APIConnectionError", None),
                                      getattr(_openai, "InternalServerError", None))
                          if isinstance(c, type) and issubclass(c, BaseException))
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=self.model, messages=messages, timeout=self.timeout,
                    max_tokens=2048, response_format={"type": "json_object"},
                )
            except transient as exc:
                # Transient gateway failure (NVIDIA returned an empty/garbled body
                # 2026-10-05 → SDK JSONDecodeError). Retry once; auth errors are not retried.
                last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                if attempt == 0:
                    continue
                raise
            choice = response.choices[0]
            raw = (choice.message.content or "").strip()
            try:
                if not raw:
                    raise ValueError(f"empty final content (finish_reason={choice.finish_reason})")
                data = self._parse_json(raw)
                advisor_resp = AdvisorResponse.from_dict({**data, "_raw": raw[:500]})
                errors = advisor_resp.validate()
                if errors:
                    raise ValueError("; ".join(errors))
                return ModelResult(success=True, data=advisor_resp,
                                   latency_ms=int((time.monotonic() - start) * 1000),
                                   snapshot_ref=snapshot_ref)
            except (ValueError, TypeError) as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                if attempt == 0:
                    messages.append({"role": "user", "content":
                        "Retry with only valid JSON: recommendation, confidence, evidence, risks. "
                        "Put all caveats in risks, with each text field at most 500 characters. "
                        "No prose or additional fields."})
        return ModelResult(success=False,
                           error=f"Advisor returned no valid JSON after retry ({last_error})",
                           latency_ms=int((time.monotonic() - start) * 1000),
                           snapshot_ref=snapshot_ref)

    def _analyze_anthropic(self, prompt: str, start: float, snapshot_ref: str) -> ModelResult:
        """Call Anthropic Claude API."""
        try:
            import anthropic
        except ImportError:
            return ModelResult(
                success=False,
                error="anthropic package not installed",
                latency_ms=int((time.monotonic() - start) * 1000),
                snapshot_ref=snapshot_ref,
            )

        api_key = self._get_api_key()
        if not api_key:
            return ModelResult(
                success=False,
                error="ADVISOR_API_KEY not set in environment",
                latency_ms=int((time.monotonic() - start) * 1000),
                snapshot_ref=snapshot_ref,
            )

        client = anthropic.Anthropic(api_key=api_key, timeout=self.timeout)

        response = client.messages.create(
            model=self.model,
            system=ADVISOR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            timeout=self.timeout,
        )

        raw = response.content[0].text.strip()
        data = self._parse_json(raw)
        advisor_resp = AdvisorResponse.from_dict({**data, "_raw": raw[:500]})

        errors = advisor_resp.validate()
        if errors:
            return ModelResult(
                success=False,
                error=f"Schema validation failed: {'; '.join(errors)}",
                latency_ms=int((time.monotonic() - start) * 1000),
                snapshot_ref=snapshot_ref,
            )

        return ModelResult(
            success=True,
            data=advisor_resp,
            latency_ms=int((time.monotonic() - start) * 1000),
            snapshot_ref=snapshot_ref,
        )

    def _analyze_mock(self, snapshot: dict, prompt: str, start: float, snapshot_ref: str) -> ModelResult:
        """Deterministic mock response based on strategy signal."""
        signal = snapshot.get("strategy_signal", 0)
        close = snapshot.get("market", {}).get("close", 0)
        sma_short = snapshot.get("market", {}).get("sma_short", 0) or 0
        sma_long = snapshot.get("market", {}).get("sma_long", 0) or 0

        if signal == 1:
            rec = "buy"
            evidence = f"SMA({int(sma_short)}) above SMA({int(sma_long)}) at close {close:.2f}; uptrend continuation signal."
            risks = "Trend may reverse; momentum-dependent. Monitor for SMA cross-back."
            confidence = 0.75
        elif signal == -1:
            rec = "sell"
            evidence = f"SMA({int(sma_short)}) below SMA({int(sma_long)}) at close {close:.2f}; downtrend continuation signal."
            risks = "Short squeeze risk; oversold bounce possible."
            confidence = 0.70
        else:
            rec = "hold"
            evidence = f"No clear SMA crossover at close {close:.2f}; SMAs near parity."
            risks = "Indecision regime; entering risks whipsaw losses."
            confidence = 0.60

        advisor_resp = AdvisorResponse(
            recommendation=rec,
            confidence=confidence,
            evidence=evidence,
            risks=risks,
        )
        return ModelResult(
            success=True,
            data=advisor_resp,
            latency_ms=int((time.monotonic() - start) * 1000),
            snapshot_ref=snapshot_ref,
        )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Parse JSON from model output, handling markdown wrappers."""
        # Strip markdown code blocks if present
        raw = raw.strip()
        if raw.startswith("```"):
            # Remove leading ```json or ```
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not a dict/object")
        return data
