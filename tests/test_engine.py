"""End-to-end + validation-layer tests for backend/engine.py.

Covers: (1) the full request -> suggestions pipeline works without the LLM
installed (the only mode testable in CI/sandbox without network access to
huggingface.co), (2) the anti-hallucination validator correctly accepts
grounded LLM output and rejects ungrounded/malformed output, (3) malformed
user input is handled without crashing ("human proof").
"""
import pytest
from pydantic import ValidationError

from backend.catalog import get_catalog
from backend.engine import _validate_llm_output, get_suggestions
from backend.schemas import SuggestRequest


def test_get_suggestions_runs_without_llm():
    req = SuggestRequest(
        skill_level="beginner",
        available_hardware=["arduino uno", "ultrasonic sensor", "buzzer"],
        interests="safety alarms",
        time_budget_hours=5,
    )
    resp = get_suggestions(req)
    assert resp.engine_mode == "template_fallback"
    assert len(resp.suggestions) == 3
    assert resp.suggestions[0].source == "template_fallback"
    assert resp.suggestions[0].rationale  # never empty


def test_suggestions_never_claim_a_missing_part_is_owned():
    req = SuggestRequest(
        skill_level="intermediate",
        available_hardware=["esp32"],
        interests="iot",
        time_budget_hours=6,
    )
    resp = get_suggestions(req)
    for s in resp.suggestions:
        assert set(s.have_components).isdisjoint(set(s.missing_components))


def test_validator_accepts_grounded_output():
    catalog = get_catalog()
    good = {
        "rationale": "This fits your beginner skill level and your interest in safety alarms nicely.",
        "buy_next": "Grab a buzzer next.",
    }
    allowed_ids = {"arduino_uno", "buzzer"}
    assert _validate_llm_output(good, allowed_ids, catalog) is True


def test_validator_rejects_hallucinated_component():
    catalog = get_catalog()
    bad = {
        "rationale": "This works great with a Raspberry Pi 5 and a robotic arm kit you didn't mention.",
        "buy_next": "Get a LoRa module.",
    }
    allowed_ids = {"arduino_uno", "buzzer"}
    assert _validate_llm_output(bad, allowed_ids, catalog) is False


def test_validator_rejects_malformed_output():
    catalog = get_catalog()
    allowed_ids = {"arduino_uno"}
    assert _validate_llm_output({"rationale": ""}, allowed_ids, catalog) is False
    assert _validate_llm_output({"rationale": "ok", "buy_next": 5}, allowed_ids, catalog) is False


# --- human-proofing: bad input never crashes the API layer ----------------

def test_invalid_skill_level_is_rejected_not_silently_accepted():
    with pytest.raises(ValidationError):
        SuggestRequest(skill_level="expert!!", available_hardware=[], interests="", time_budget_hours=5)


def test_negative_time_budget_is_clamped_not_crashing():
    req = SuggestRequest(skill_level="beginner", available_hardware=[], interests="", time_budget_hours=-5)
    assert req.time_budget_hours > 0
    resp = get_suggestions(req)
    assert len(resp.suggestions) == 3


def test_absurdly_large_time_budget_is_capped():
    req = SuggestRequest(skill_level="beginner", available_hardware=[], interests="", time_budget_hours=10_000_000)
    assert req.time_budget_hours <= 200


def test_empty_everything_still_returns_suggestions():
    req = SuggestRequest(skill_level="beginner", available_hardware=[], interests="", time_budget_hours=4)
    resp = get_suggestions(req)
    assert len(resp.suggestions) == 3
    for s in resp.suggestions:
        assert s.title
