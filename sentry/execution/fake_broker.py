"""In-memory simulated broker for tests and the end-to-end dry run. Never touches a network.

Market orders queue on submit() and fill on fill_open_orders() at the supplied price with
CostModel.per_side applied to the fill price (buy at p * (1 + c), sell at p * (1 - c)), so the
trade log P&L and the broker equity agree to the cent, as they would with real slippage.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Mapping, Optional

from sentry.config import CostModel
from sentry.execution.broker import BrokerAccount, Fill
from sentry.types import Order, Position


class FakeBroker:
    is_paper = True

    def __init__(
        self, cash: float, costs: CostModel, trading_days: Iterable[date], prices: Optional[Mapping[str, float]] = None
    ):
        self.cash = float(cash)
        self.costs = costs
        self._days = sorted(trading_days)
        self._prices: dict[str, float] = dict(prices or {})
        self._pos: dict[str, tuple[float, float]] = {}  # symbol -> (signed qty, avg entry price)
        self._open: dict[str, Order] = {}  # broker order id -> order
        self._seen_client_ids: set[str] = set()
        self._fills: list[Fill] = []
        self._history: list[tuple[datetime, float]] = []
        self._last_equity: Optional[float] = None
        self._n = 0

    # ---- market data
    def set_prices(self, prices: Mapping[str, float]) -> None:
        self._prices.update(prices)

    def latest_prices(self, symbols: list[str]) -> dict[str, float]:
        return {s: self._prices[s] for s in symbols if s in self._prices}

    def calendar(self, start: date, end: date) -> list[date]:
        return [d for d in self._days if start <= d <= end]

    # ---- account
    def equity(self) -> float:
        return self.cash + sum(q * self._prices[s] for s, (q, _) in self._pos.items())

    def account(self) -> BrokerAccount:
        return BrokerAccount(self.equity(), self.cash, self._last_equity, tuple(self._history), is_paper=True)

    def positions(self) -> dict[str, Position]:
        return {s: Position(s, q, avg, q * self._prices[s]) for s, (q, avg) in self._pos.items()}

    def mark_close(self, at: datetime) -> float:
        """End-of-day equity point; becomes last_equity (the next day's day_start_equity)."""
        eq = self.equity()
        self._history.append((at, eq))
        self._last_equity = eq
        return eq

    # ---- orders
    def submit(self, order: Order) -> str:
        if order.client_order_id in self._seen_client_ids:
            raise ValueError(f"duplicate client_order_id {order.client_order_id}")  # Alpaca rejects these too
        if (order.qty is None) == (order.notional is None):
            raise ValueError("exactly one of qty / notional")
        self._seen_client_ids.add(order.client_order_id)
        self._n += 1
        oid = f"fake-{self._n}"
        self._open[oid] = order
        return oid

    def open_orders(self) -> list[Order]:
        return list(self._open.values())

    def cancel(self, order_id: str) -> None:
        self._open.pop(order_id, None)

    def fills(self, after: Optional[datetime] = None) -> list[Fill]:
        return [f for f in self._fills if after is None or f.filled_at > after]

    def fill_open_orders(self, at: datetime, prices: Optional[Mapping[str, float]] = None) -> list[Fill]:
        if prices:
            self.set_prices(prices)
        c = self.costs.per_side
        new: list[Fill] = []
        for oid in sorted(self._open, key=lambda k: int(k.split("-")[1])):
            o = self._open.pop(oid)
            sgn = 1 if o.side == "buy" else -1
            fp = self._prices[o.symbol] * (1 + sgn * c)
            qty = o.qty if o.qty is not None else o.notional / fp
            self._apply(o.symbol, sgn * qty, fp)
            self.cash -= sgn * qty * fp
            f = Fill(oid, o.client_order_id, o.symbol, o.side, qty, fp, at)
            self._fills.append(f)
            new.append(f)
        return new

    def _apply(self, sym: str, dq: float, fp: float) -> None:
        q, avg = self._pos.get(sym, (0.0, 0.0))
        new = q + dq
        if abs(new) < 1e-12:
            self._pos.pop(sym, None)
            return
        if q == 0 or (q > 0) == (dq > 0):
            avg = (q * avg + dq * fp) / new  # opening or adding: average cost
        elif (new > 0) != (q > 0):
            avg = fp  # flipped through zero: remainder opened at this fill
        self._pos[sym] = (new, avg)
