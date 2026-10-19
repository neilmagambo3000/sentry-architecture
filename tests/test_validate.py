import pydantic
import pytest

from sentry.agents.context import ContextBundle
from sentry.agents.schema import SCHEMA_VERSION, AgentView
from sentry.agents.validate import EvidenceLeakError, SchemaDriftError, validate

VIEW = dict(direction=1, confidence=0.5, horizon_days=5, key_drivers=["x"], invalidators=["y"], event_risk="none",
            rationale="short", evidence_ids=["a1", "a2"])
BUNDLE = ContextBundle("s", "d", "c", frozenset({"a1", "a2", "a3"}), "h", False)


def test_subset_passes():
    v = AgentView(**VIEW)
    assert validate(v, BUNDLE) is v
    assert validate(AgentView(**{**VIEW, "evidence_ids": []}), BUNDLE)


def test_evidence_leak_raises():
    with pytest.raises(EvidenceLeakError, match="zz"):
        validate(AgentView(**{**VIEW, "evidence_ids": ["a1", "zz"]}), BUNDLE)


def test_schema_drift_raises():
    with pytest.raises(SchemaDriftError):
        validate(AgentView(**VIEW), BUNDLE, schema_version=SCHEMA_VERSION + "-old")


@pytest.mark.parametrize("bad", [
    {"direction": 2}, {"confidence": 1.1}, {"horizon_days": 3}, {"key_drivers": ["a"] * 4},
    {"key_drivers": ["x" * 161]}, {"invalidators": ["a"] * 3}, {"event_risk": "war"},
    {"rationale": "x" * 481}, {"evidence_ids": [str(i) for i in range(9)]}, {"symbol": "AAA"},
])
def test_agentview_limits(bad):
    with pytest.raises(pydantic.ValidationError):
        AgentView(**{**VIEW, **bad})


def test_json_schema_forbids_extra():
    assert AgentView.model_json_schema()["additionalProperties"] is False
