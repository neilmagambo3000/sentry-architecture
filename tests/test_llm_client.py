"""LLM client contract tests. A fake client returns synthetic fixtures; the real Anthropic client is never built."""
import json
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import pandas as pd
import pytest

from sentry.agents.context import ContextBundle
from sentry.agents.disc import run_disc
from sentry.agents.schema import AgentView
from sentry.agents.tagger import KeywordTagger
from sentry.llm import client as client_mod
from sentry.llm.client import call
from sentry.llm.cost import CostBudget, CostCeilingError
from sentry.llm.registry import PromptRegistry, UnregisteredPromptError
from sentry.types import Article, Snapshot

FIX = Path(__file__).parent / "fixtures" / "llm"
PRICES = {"input": 2.0, "output": 10.0, "cache_write": 2.5, "cache_read": 0.2}
AS_OF = datetime(2026, 9, 30, 12, tzinfo=timezone.utc)


class FakeClient:
    def __init__(self, fixture=None, exc=None):
        self.fixture, self.exc, self.calls = fixture, exc, []
        self.messages = self

    def parse(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        d = json.loads((FIX / self.fixture).read_text())
        assert d["synthetic"] is True
        out = AgentView(**d["parsed_output"]) if d["parsed_output"] else None
        return SimpleNamespace(stop_reason=d["stop_reason"], usage=SimpleNamespace(**d["usage"]), parsed_output=out)


@pytest.fixture(autouse=True)
def no_real_client(monkeypatch):
    def boom():
        raise AssertionError("real Anthropic client must never be built in tests")
    monkeypatch.setattr(client_mod, "_real_client", boom)


@pytest.fixture
def dev(monkeypatch):
    monkeypatch.setenv("SENTRY_ENV", "dev")


@pytest.fixture
def registry(tmp_path):
    return PromptRegistry(tmp_path / "registry.json")


def bundle(ids=("a1",)):
    return ContextBundle(system="S" * 5000, daily_digest="digest", symbol_context="ctx",
                         visible_ids=frozenset(ids), context_hash="h1", masked=False)


def run(db, registry, fake, budget_usd=10.0, **kw):
    budget = CostBudget(db, budget_usd, date(2026, 9, 30), prices=PRICES)
    return call("disc", kw.pop("b", bundle()), symbol="AAA", as_of=AS_OF, model_id="claude-sonnet-5",
                temperature=1.0, effort="low", schema=AgentView, budget=budget, registry=registry,
                conn=db, client=fake, **kw), budget


def test_ok_call_logs_meters_cost_and_shape(db, registry, dev):
    fake = FakeClient("synthetic_disc_ok.json")
    c, budget = run(db, registry, fake)
    assert c.view is not None and c.error is None and not c.cache_hit and c.frozen is False
    expected = (900 * 2.0 + 2100 * 2.5 + 0 * 0.2 + 180 * 10.0) / 1e6
    assert c.cost_usd == pytest.approx(expected) and budget.spent_today() == pytest.approx(expected)
    row = db.execute("select * from agent_calls").fetchone()
    assert row["cache_creation_input_tokens"] == 2100 and row["output_tokens"] == 180 and row["frozen"] == 0
    kw = fake.calls[0]
    assert "temperature" not in kw
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in kw["messages"][0]["content"][1]
    assert kw["output_format"] is AgentView and kw["output_config"] == {"effort": "low"}


def test_cache_hit_skips_client_and_costs_nothing(db, registry, dev):
    fake = FakeClient("synthetic_disc_ok.json")
    first, budget = run(db, registry, fake)
    second, _ = run(db, registry, fake)
    assert len(fake.calls) == 1
    assert second.cache_hit and second.cost_usd == 0 and second.view == first.view
    assert budget.spent_today() == pytest.approx(first.cost_usd)


def test_cost_ceiling_raises_before_client(db, registry, dev):
    fake = FakeClient("synthetic_disc_ok.json")
    with pytest.raises(CostCeilingError):
        run(db, registry, fake, budget_usd=0.001)
    assert len(fake.calls) == 0


def test_prod_refuses_unfrozen_prompt(db, registry, monkeypatch):
    monkeypatch.delenv("SENTRY_ENV", raising=False)
    fake = FakeClient("synthetic_disc_ok.json")
    with pytest.raises(UnregisteredPromptError):
        run(db, registry, fake)
    assert len(fake.calls) == 0


def test_prod_allows_frozen_prompt(db, registry, dev, monkeypatch):
    c, _ = run(db, registry, FakeClient("synthetic_disc_ok.json"), use_cache=False)
    registry.freeze(c.prompt_hash, "disc", "2026-09-30")
    monkeypatch.delenv("SENTRY_ENV")
    c2, _ = run(db, registry, FakeClient("synthetic_disc_ok.json"))
    assert c2.frozen is True and c2.view is not None and not c2.cache_hit


@pytest.mark.parametrize("fixture,reason", [("synthetic_disc_refusal.json", "refusal"),
                                            ("synthetic_disc_max_tokens.json", "max_tokens")])
def test_refusal_and_max_tokens_give_error_no_view_not_cached(db, registry, dev, fixture, reason):
    fake = FakeClient(fixture)
    c, budget = run(db, registry, fake)
    assert c.view is None and reason in c.error and c.cost_usd > 0
    assert budget.spent_today() == pytest.approx(c.cost_usd)
    run(db, registry, fake)
    assert len(fake.calls) == 2  # not cached


def test_sdk_errors_become_logged_error_calls(db, registry, dev):
    req = httpx2.Request("POST", "https://example.invalid/v1/messages")
    for exc in (anthropic.RateLimitError("rate", response=httpx2.Response(429, request=req), body=None),
                anthropic.APIStatusError("boom", response=httpx2.Response(500, request=req), body=None),
                anthropic.APIConnectionError(request=req)):
        c, _ = run(db, registry, FakeClient(exc=exc))
        assert c.view is None and type(exc).__name__ in c.error and c.cost_usd == 0
        assert db.execute("select error from agent_calls").fetchone()[0] == c.error


def snapshot():
    arts = (Article("a1", ("AAA",), AS_OF.replace(hour=9), None, "Reuters", "AAA raises guidance", None, "u"),)
    return Snapshot("AAA", AS_OF, pd.DataFrame(), arts, (), {})


def test_run_disc_valid_view(db, registry, dev, settings):
    c = run_disc(snapshot(), "ta", "fa", "digest", tagger=KeywordTagger(), masked=False,
                 models=settings.models, budget=CostBudget(db, 10.0, date(2026, 9, 30), prices=PRICES),
                 registry=registry, conn=db, client=FakeClient("synthetic_disc_ok.json"))
    assert c.view is not None and c.view.evidence_ids == ["a1"]


def test_run_disc_drops_evidence_leak(db, registry, dev, settings):
    c = run_disc(snapshot(), "ta", "fa", "digest", tagger=KeywordTagger(), masked=False,
                 models=settings.models, budget=CostBudget(db, 10.0, date(2026, 9, 30), prices=PRICES),
                 registry=registry, conn=db, client=FakeClient("synthetic_disc_evidence_leak.json"))
    assert c.view is None and "EvidenceLeakError" in c.error
    row = db.execute("select error, view_json from agent_calls").fetchone()
    assert "EvidenceLeakError" in row["error"] and row["view_json"] is None
