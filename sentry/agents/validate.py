"""Post-validation."""
from __future__ import annotations

from sentry.agents.context import ContextBundle
from sentry.agents.schema import SCHEMA_VERSION, AgentView


class EvidenceLeakError(ValueError):
    pass


class SchemaDriftError(ValueError):
    pass


def validate(view: AgentView, bundle: ContextBundle, schema_version: str = SCHEMA_VERSION) -> AgentView:
    """schema_version is the one stamped on the AgentCall; AgentView itself carries none (extra=forbid)."""
    if schema_version != SCHEMA_VERSION:
        raise SchemaDriftError(f"schema_version {schema_version!r} != {SCHEMA_VERSION!r}")
    leaked = set(view.evidence_ids) - bundle.visible_ids
    if leaked:
        raise EvidenceLeakError(f"evidence_ids not in visible set: {sorted(leaked)}")
    return view
