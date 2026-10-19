"""Shared data contracts. Every module builds against these.

Pure dataclasses: no I/O, no LLM, no clock. `AgentView` / `AgentCall` live in
`sentry/agents/schema.py` because they belong to Layer B.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal, Mapping, Optional

import pandas as pd

Direction = Literal[-1, 0, 1]

# ---------------------------------------------------------------- data layer (§2.1)


@dataclass(frozen=True)
class Article:
    id: str
    symbols: tuple[str, ...]
    first_seen: datetime  # tz-aware UTC; live = arrival time, history = vendor publish time
    published: Optional[datetime]
    source: str
    headline: str
    body: Optional[str]
    url: str


@dataclass(frozen=True)
class Filing:
    id: str
    symbol: str
    form: str  # "10-Q" | "10-K" | ...
    filed_at: datetime  # tz-aware UTC
    period_end: date
    facts: Mapping[str, float]  # first-reported XBRL facts only, never restated


@dataclass(frozen=True)
class MacroVintage:
    series: str
    vintage: datetime  # tz-aware UTC: when this vintage became public
    observations: Mapping[date, float]


@dataclass(frozen=True)
class Snapshot:
    symbol: str
    as_of: datetime
    bars: pd.DataFrame  # index: date (ascending); columns: open high low close volume adj_close
    articles: tuple[Article, ...]
    filings: tuple[Filing, ...]
    macro: Mapping[str, MacroVintage]


# ---------------------------------------------------------------- Layer A / risk / execution (§7, §9, §10)


@dataclass(frozen=True)
class Decision:
    symbol: str
    as_of: datetime
    direction: int  # -1, 0, +1
    target_weight: float  # signed fraction of equity
    stop: Optional[float]  # price level, None when flat
    score: float
    reason: Mapping[str, object]
    params_hash: str
    strategy_version: str
    strategy_hash: str


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float  # signed; fractional allowed
    avg_entry_price: float
    market_value: float
    opened_at: Optional[datetime] = None  # from our trade log; broker does not report it


@dataclass(frozen=True)
class ExitSignal:
    exit: bool
    reason: str  # "stop" | "time" | "flip" | "none"


@dataclass(frozen=True)
class AccountState:
    """Built ONLY by execution/reconcile.py from broker-reported data (§9)."""

    as_of: datetime
    equity: float
    cash: float
    positions: Mapping[str, Position]
    equity_history: tuple[tuple[datetime, float], ...]  # broker-reported, ascending
    day_start_equity: float
    peak_equity: float
    is_paper: bool


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Literal["buy", "sell"]
    notional: Optional[float]  # USD, for fractional/notional market orders
    qty: Optional[float]  # used to close a whole position
    client_order_id: str
    strategy_version: str
    strategy_hash: str
    type: Literal["market"] = "market"
    time_in_force: Literal["day"] = "day"
    reason: str = ""


@dataclass(frozen=True)
class GateResult:
    approved: tuple[Decision, ...]
    halt_all: bool
    no_new_entries: bool
    flatten: bool
    reasons: tuple[str, ...]  # one logged line per check that fired


@dataclass(frozen=True)
class EventCalendar:
    earnings_in_days: Mapping[str, Optional[int]] = field(default_factory=dict)
