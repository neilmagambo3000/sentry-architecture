"""AgentView (the structured-output schema) and AgentCall (one logged call)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

SCHEMA_VERSION = "agentview-v1"

Short = Annotated[str, StringConstraints(max_length=160)]


class AgentView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction: Literal[-1, 0, 1]
    confidence: float = Field(ge=0.0, le=1.0)
    horizon_days: Literal[1, 5, 20]
    key_drivers: list[Short] = Field(max_length=3)
    invalidators: list[Short] = Field(max_length=2)
    event_risk: Literal["none", "earnings", "filing", "macro", "other"]
    rationale: str = Field(max_length=480)
    evidence_ids: list[str] = Field(max_length=8)


AgentName = Literal["digest", "disc", "trader", "bull", "bear", "risk_reviewer", "lap_probe"]


@dataclass(frozen=True)
class AgentCall:
    agent: AgentName
    symbol: str
    as_of: datetime
    model_id: str
    prompt_hash: str
    schema_version: str
    temperature: float
    input_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    output_tokens: int
    cost_usd: float
    cache_hit: bool
    batch_id: Optional[str]
    escalated: bool
    view: Optional[AgentView]
    error: Optional[str]
    latency_ms: int
    # Not in DESIGN §2.2's list; mirrors agent_calls.frozen (False only for SENTRY_ENV=dev calls, §6.3).
    frozen: bool = True
