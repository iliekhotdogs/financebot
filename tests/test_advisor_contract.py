"""Advisor final-content contract and safe forwarding to Jev."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from advisor import AdvisorClient
from jev import JevAgent
from schemas import AdvisorResponse


CONFIG = {"advisor": {"enabled": True, "provider": "openai", "model": "moonshotai/kimi-k3", "api_key_env": "TEST_ADVISOR_KEY"}}
SNAP = {"snapshot_id": "test", "timestamp": "2026-10-05T16:00:00+00:00", "symbol": "TXN", "market": {}, "account": {}, "position": None, "strategy_signal": 1, "strategy_name": "test", "valid_actions": ["buy", "hold"]}


def reply(content, reasoning="unstructured internal analysis"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, reasoning_content=reasoning), finish_reason="stop")])


def run_with_replies(monkeypatch, replies):
    monkeypatch.setenv("TEST_ADVISOR_KEY", "fake")
    with patch("openai.OpenAI") as factory:
        create = factory.return_value.chat.completions.create
        create.side_effect = replies
        result = AdvisorClient(CONFIG).analyze(SNAP)
        return result, create.call_args_list


def test_retries_empty_final_and_forwards_every_field(monkeypatch):
    obj = {"recommendation": "hold", "confidence": 0.7, "evidence": "Trend weakening", "risks": "Gap risk; avoid entry"}
    result, calls = run_with_replies(monkeypatch, [reply(None), reply(json.dumps(obj))])
    assert result.success and len(calls) == 2
    assert all(c.kwargs["response_format"] == {"type": "json_object"} for c in calls)
    prompt = JevAgent({"jev": {}})._build_prompt(SNAP, result)
    for text in ("hold", "0.70", obj["evidence"], obj["risks"]):
        assert text in prompt


def test_rejects_internal_reasoning_and_malformed_retries(monkeypatch):
    result, calls = run_with_replies(monkeypatch, [reply(None, '{"recommendation":"buy"}'), reply("not json")])
    assert not result.success and len(calls) == 2
    assert "after retry" in result.error


@pytest.mark.parametrize("field,value", [("risks", "x" * 501), ("evidence", ""), ("extra", "unforwarded warning")])
def test_no_silent_loss(field, value):
    payload = {"recommendation": "hold", "confidence": 0.5, "evidence": "a", "risks": "b"}
    payload[field] = value
    with pytest.raises(ValueError):
        AdvisorResponse.from_dict(payload)
