import json

import pytest

from sentry.config import ROOT
from sentry.llm.registry import PromptRegistry, UnregisteredPromptError, prompt_hash

BASE = dict(agent="disc", system_text="sys", schema_version="agentview-v1", temperature=1.0,
            model_id="claude-sonnet-5", gate_yaml_bytes=b"news_lag_minutes: 30\n", masked=False)


def test_prompt_hash_stable_and_sensitive_to_every_input():
    h = prompt_hash(**BASE)
    assert h == prompt_hash(**BASE) and len(h) == 64
    for k, v in [("agent", "trader"), ("system_text", "sys2"), ("schema_version", "v2"), ("temperature", 0.5),
                 ("model_id", "claude-opus-5"), ("gate_yaml_bytes", b"news_lag_minutes: 31\n"), ("masked", True)]:
        assert prompt_hash(**{**BASE, k: v}) != h, k


def test_repo_registry_starts_empty():
    assert json.loads((ROOT / "config" / "registry.json").read_text()) == {"frozen": {}}


def test_freeze_persists(tmp_path):
    p = tmp_path / "registry.json"
    r = PromptRegistry(p)
    assert not r.is_frozen("abc")
    r.freeze("abc", "disc", "2026-09-30")
    assert PromptRegistry(p).is_frozen("abc")
    assert json.loads(p.read_text())["frozen"]["abc"] == {"agent": "disc", "frozen_on": "2026-09-30"}


def test_prod_refuses_unfrozen_dev_allows_but_marks(tmp_path, monkeypatch):
    r = PromptRegistry(tmp_path / "registry.json")
    monkeypatch.delenv("SENTRY_ENV", raising=False)
    with pytest.raises(UnregisteredPromptError):
        r.check("abc")
    monkeypatch.setenv("SENTRY_ENV", "dev")
    assert r.check("abc") is False
    r.freeze("abc", "disc", "2026-09-30")
    monkeypatch.delenv("SENTRY_ENV")
    assert r.check("abc") is True
