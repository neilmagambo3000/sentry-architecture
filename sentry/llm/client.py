"""The one synchronous LLM call path.

Order: registry check, cost ceiling (raises before network), cache lookup, parse call,
refusal/max_tokens handling, agent_calls log + cost record, cache store.
The real Anthropic client is built lazily from env only when no client is injected.
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from typing import TYPE_CHECKING

import anthropic
import pydantic

from sentry.agents.schema import SCHEMA_VERSION, AgentCall, AgentView
from sentry.config import config_path
from sentry.llm import cache
from sentry.llm.cost import CostBudget, CostCeilingError, estimate_usd, usage_usd
from sentry.llm.registry import PromptRegistry, prompt_hash

if TYPE_CHECKING:
    from sentry.agents.context import ContextBundle


def _real_client():
    # Reads ANTHROPIC_API_KEY from the environment; never logged. Retries only via the SDK.
    return anthropic.Anthropic(max_retries=3)


def _log(conn: sqlite3.Connection, c: AgentCall, ablation_key: str) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO agent_calls (agent, symbol, as_of, prompt_hash, ablation_key, model_id,
        schema_version, temperature, input_tokens, cache_creation_input_tokens, cache_read_input_tokens,
        output_tokens, cost_usd, cache_hit, batch_id, escalated, view_json, error, latency_ms, frozen)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (c.agent, c.symbol, c.as_of.isoformat(), c.prompt_hash, ablation_key, c.model_id, c.schema_version,
         c.temperature, c.input_tokens, c.cache_creation_input_tokens, c.cache_read_input_tokens,
         c.output_tokens, c.cost_usd, int(c.cache_hit), c.batch_id, int(c.escalated),
         c.view.model_dump_json() if c.view else None, c.error, c.latency_ms, int(c.frozen)))
    conn.commit()


def call(agent: str, bundle: "ContextBundle", *, symbol: str, as_of: datetime, model_id: str,
         temperature: float, effort: str, schema: type[AgentView], budget: CostBudget,
         registry: PromptRegistry, conn: sqlite3.Connection, client=None, ablation_key: str = "v1",
         max_tokens: int = 1024, use_cache: bool = True, gate_yaml_bytes: bytes | None = None) -> AgentCall:
    """use_cache=False is for the stability check only: it neither reads nor writes the cache."""
    gate_bytes = gate_yaml_bytes if gate_yaml_bytes is not None else config_path("gate.yaml").read_bytes()
    ph = prompt_hash(agent, bundle.system, SCHEMA_VERSION, temperature, model_id, gate_bytes, bundle.masked)

    def result(**kw) -> AgentCall:
        base = dict(agent=agent, symbol=symbol, as_of=as_of, model_id=model_id, prompt_hash=ph,
                    schema_version=SCHEMA_VERSION, temperature=temperature, input_tokens=0,
                    cache_creation_input_tokens=0, cache_read_input_tokens=0, output_tokens=0, cost_usd=0.0,
                    cache_hit=False, batch_id=None, escalated=False, view=None, error=None, latency_ms=0,
                    frozen=frozen)
        base.update(kw)
        return AgentCall(**base)

    # (1) registry: prod raises UnregisteredPromptError, dev returns False and rows are marked unfrozen
    frozen = registry.check(ph)

    # (2) cost ceiling, before any network
    est = estimate_usd(len(bundle.system) + len(bundle.daily_digest) + len(bundle.symbol_context),
                       max_tokens, budget.prices)
    if budget.would_exceed(est):
        raise CostCeilingError(f"estimate ${est:.4f} would exceed ${budget.daily_usd:.2f}/day "
                               f"(spent ${budget.spent_today():.4f})")

    # (3) cache: a hit costs $0 and never touches the client; the original agent_calls row stands
    key = cache.cache_key(agent, model_id, ph, SCHEMA_VERSION, symbol, as_of, bundle.context_hash)
    if use_cache:
        hit = cache.get(conn, key)
        if hit is not None:
            return result(view=schema.model_validate_json(hit), cache_hit=True)

    # (4) the call
    client = client if client is not None else _real_client()
    t0 = time.monotonic()
    try:
        resp = client.messages.parse(
            model=model_id, max_tokens=max_tokens,
            system=[{"type": "text", "text": bundle.system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": [
                {"type": "text", "text": bundle.daily_digest, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": bundle.symbol_context}]}],
            output_format=schema, output_config={"effort": effort})
    except (anthropic.APIStatusError, anthropic.APIConnectionError, pydantic.ValidationError) as e:
        c = result(error=f"{type(e).__name__}: {e}"[:500], latency_ms=int((time.monotonic() - t0) * 1000))
        _log(conn, c, ablation_key)
        return c
    latency = int((time.monotonic() - t0) * 1000)

    # (5) refusal / truncation: tokens were spent, so meters and cost are still logged, but no view
    u = resp.usage
    usd = usage_usd(u, budget.prices)
    meters = dict(input_tokens=u.input_tokens, cache_creation_input_tokens=u.cache_creation_input_tokens or 0,
                  cache_read_input_tokens=u.cache_read_input_tokens or 0, output_tokens=u.output_tokens,
                  cost_usd=usd, latency_ms=latency)
    view, error = None, None
    if resp.stop_reason in ("refusal", "max_tokens"):
        error = f"stop_reason={resp.stop_reason}"
    elif resp.parsed_output is None:
        error = f"no parsed output (stop_reason={resp.stop_reason})"
    else:
        view = resp.parsed_output

    # (6) log + record cost
    c = result(view=view, error=error, **meters)
    _log(conn, c, ablation_key)
    budget.record(usd)

    # (7) cache only good views
    if view is not None and use_cache:
        cache.put(conn, key, view.model_dump_json())
    return c

