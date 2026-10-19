"""Discretionary agent: build_bundle -> call -> validate.

An invalid view is logged with its error and dropped (view=None); the caller then carries the
prior view with its age, never a fresh unvalidated one.
"""
from __future__ import annotations

import dataclasses
import sqlite3

from sentry.agents.context import build_bundle
from sentry.agents.schema import AgentCall, AgentView
from sentry.agents.tagger import Tagger
from sentry.agents.validate import EvidenceLeakError, SchemaDriftError, validate
from sentry.config import ModelsConfig
from sentry.llm.client import call
from sentry.llm.cost import CostBudget
from sentry.llm.registry import PromptRegistry
from sentry.types import Snapshot


def run_disc(snapshot: Snapshot, ta_text: str, fa_text: str, daily_digest: str, *, tagger: Tagger,
             masked: bool, models: ModelsConfig, budget: CostBudget, registry: PromptRegistry,
             conn: sqlite3.Connection, client=None, ablation_key: str = "v1") -> AgentCall:
    bundle = build_bundle(snapshot, ta_text, fa_text, daily_digest, tagger, masked)
    c = call("disc", bundle, symbol=snapshot.symbol, as_of=snapshot.as_of, model_id=models.model_id,
             temperature=models.temperature, effort=models.effort, schema=AgentView, budget=budget,
             registry=registry, conn=conn, client=client, ablation_key=ablation_key,
             max_tokens=models.max_tokens)
    if c.view is None:
        return c
    try:
        validate(c.view, bundle, c.schema_version)
    except (EvidenceLeakError, SchemaDriftError) as e:
        c = dataclasses.replace(c, view=None, error=f"{type(e).__name__}: {e}"[:500])
        if not c.cache_hit:  # a cache hit has no row of its own; the original row was already marked
            conn.execute("UPDATE agent_calls SET error = ?, view_json = NULL WHERE agent = ? AND symbol = ? "
                         "AND as_of = ? AND prompt_hash = ? AND ablation_key = ?",
                         (c.error, c.agent, c.symbol, c.as_of.isoformat(), c.prompt_hash, ablation_key))
            conn.commit()
    return c
