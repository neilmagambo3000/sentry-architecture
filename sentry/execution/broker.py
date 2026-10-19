"""Broker protocol.

Deviation from the DESIGN sketch: account() returns a BrokerAccount (raw broker numbers), not
an AccountState, because AccountState is built only by execution/reconcile.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Optional, Protocol

from sentry.types import Order, Position


@dataclass(frozen=True)
class BrokerAccount:
    equity: float
    cash: float
    last_equity: Optional[float]  # broker-reported equity at the previous trading day's close
    equity_history: tuple[tuple[datetime, float], ...]  # broker-reported, ascending
    is_paper: bool


@dataclass(frozen=True)
class Fill:
    broker_order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: Literal["buy", "sell"]
    qty: float  # unsigned
    price: float
    filled_at: datetime


class Broker(Protocol):
    def account(self) -> BrokerAccount: ...
    def positions(self) -> dict[str, Position]: ...
    def submit(self, order: Order) -> str: ...  # returns the broker order id
    def open_orders(self) -> list[Order]: ...
    def cancel(self, order_id: str) -> None: ...
    def fills(self, after: Optional[datetime] = None) -> list[Fill]: ...
    def calendar(self, start: date, end: date) -> list[date]: ...
    def latest_prices(self, symbols: list[str]) -> dict[str, float]: ...
