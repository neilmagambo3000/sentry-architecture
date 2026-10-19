"""Alpaca PAPER broker over alpaca-py 0.44.0. Paper only, enforced in code.

- The real TradingClient is only ever built in from_env(), always with paper=True. There is no
  argument anywhere in this class that selects a non-paper account or a different base URL.
- The constructor rejects any client whose resolved base URL is not PAPER_URL, and any key id
  that does not carry Alpaca's paper prefix "PK".
- Dry run (SENTRY_DRY_RUN, ON unless set to an explicit false token: 0/false/no/off): submit() and
  cancel() never reach the client.
- Keys are read from ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET in from_env() only and are never
  stored on the instance or logged.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Any, Optional

from sentry.config import env_flag
from sentry.execution.broker import BrokerAccount, Fill
from sentry.types import Order, Position

PAPER_URL = "https://paper-api.alpaca.markets"
PAPER_KEY_PREFIX = "PK"


class LiveEndpointError(RuntimeError):
    pass


class DryRunError(RuntimeError):
    pass


def _resolved_base_url(client: Any) -> str:
    url = getattr(client, "_base_url", None)
    return str(getattr(url, "value", url))


def _f(x: Any) -> Optional[float]:
    return None if x is None else float(x)


def _enum(x: Any) -> str:
    return str(getattr(x, "value", x))


class AlpacaPaperBroker:
    is_paper = True

    def __init__(self, client: Any, key_id: str, *, data_client: Any = None, dry_run: Optional[bool] = None):
        if not isinstance(key_id, str) or not key_id.startswith(PAPER_KEY_PREFIX):
            raise LiveEndpointError("Alpaca key id is not a paper key (expected prefix 'PK'); refusing to start")
        if _resolved_base_url(client) != PAPER_URL:
            raise LiveEndpointError("trading client is not pointed at the Alpaca paper endpoint; refusing to start")
        if getattr(client, "_sandbox", False) is not True:
            raise LiveEndpointError("trading client is not in paper (sandbox) mode; refusing to start")
        self._client = client
        self._data = data_client
        self.dry_run = env_flag("SENTRY_DRY_RUN", default=True) if dry_run is None else bool(dry_run)

    @classmethod
    def from_env(cls) -> "AlpacaPaperBroker":
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        key_id = os.environ["ALPACA_PAPER_KEY_ID"]
        secret = os.environ["ALPACA_PAPER_SECRET"]
        if not key_id.startswith(PAPER_KEY_PREFIX):
            raise LiveEndpointError("ALPACA_PAPER_KEY_ID is not a paper key (expected prefix 'PK')")
        client = TradingClient(api_key=key_id, secret_key=secret, paper=True)
        return cls(client, key_id, data_client=StockHistoricalDataClient(key_id, secret))

    # ---- account
    def account(self) -> BrokerAccount:
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        a = self._client.get_account()
        h = self._client.get_portfolio_history(GetPortfolioHistoryRequest(period="1A", timeframe="1D"))
        hist = tuple(
            (datetime.fromtimestamp(ts, tz=timezone.utc), float(eq))
            for ts, eq in zip(h.timestamp, h.equity, strict=False)
            if eq is not None and float(eq) > 0
        )
        # Paper account numbers start with "PA"; anything else is reported as not paper so the
        # risk gate halts everything.
        is_paper = str(getattr(a, "account_number", "")).startswith("PA")
        return BrokerAccount(float(a.equity), float(a.cash), _f(a.last_equity), hist, is_paper)

    def positions(self) -> dict[str, Position]:
        out = {}
        for p in self._client.get_all_positions():
            q = abs(float(p.qty))
            if _enum(p.side) == "short":
                q = -q
            out[p.symbol] = Position(p.symbol, q, float(p.avg_entry_price), float(p.market_value or 0.0))
        return out

    # ---- orders
    def submit(self, order: Order) -> str:
        if self.dry_run:
            raise DryRunError("dry run is on: orders are logged by submit_orders, never sent")
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        req = MarketOrderRequest(
            symbol=order.symbol,
            notional=order.notional,
            qty=order.qty,
            side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=order.client_order_id,
        )
        return str(self._client.submit_order(req).id)

    def open_orders(self) -> list[Order]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        return [
            Order(
                symbol=o.symbol, side=_enum(o.side), notional=_f(o.notional), qty=_f(o.qty),
                client_order_id=o.client_order_id, strategy_version="", strategy_hash="",
            )
            for o in self._client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
        ]

    def cancel(self, order_id: str) -> None:
        if self.dry_run:
            raise DryRunError("dry run is on: nothing is sent to the broker, not even a cancel")
        self._client.cancel_order_by_id(order_id)

    def fills(self, after: Optional[datetime] = None) -> list[Fill]:
        """Aggregated per order (filled_qty at filled_avg_price), read back from the broker."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        req = GetOrdersRequest(status=QueryOrderStatus.ALL, after=after, limit=500)
        return [
            Fill(str(o.id), o.client_order_id, o.symbol, _enum(o.side), float(o.filled_qty), float(o.filled_avg_price), o.filled_at)
            for o in self._client.get_orders(req)
            if o.filled_at is not None and o.filled_qty and float(o.filled_qty) > 0
        ]

    # ---- calendar / prices
    def calendar(self, start: date, end: date) -> list[date]:
        from alpaca.trading.requests import GetCalendarRequest

        return [c.date for c in self._client.get_calendar(GetCalendarRequest(start=start, end=end))]

    def latest_prices(self, symbols: list[str]) -> dict[str, float]:
        if self._data is None or not symbols:
            return {}
        from alpaca.data.requests import StockLatestTradeRequest

        trades = self._data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=list(symbols)))
        return {s: float(t.price) for s, t in trades.items()}
