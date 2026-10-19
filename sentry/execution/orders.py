"""Approved target weights -> broker orders, and idempotent submission.

Full closes use qty (the broker-reported position size); everything else is a notional market
DAY order (fractional, the default order type). A flip (target on the other side of the
current position) only closes today; any re-entry comes from a later cycle, so one symbol never
gets a close and an opposite entry in the same cycle.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Mapping, Sequence

from sentry.types import AccountState, Decision, Order

log = logging.getLogger(__name__)


def client_order_id(strategy_hash: str, symbol: str, as_of: datetime, side: str) -> str:
    return f"sentry-{strategy_hash[:8]}-{symbol}-{as_of:%Y%m%d}-{side}"


def to_orders(
    approved: Sequence[Decision],
    account: AccountState,
    prices: Mapping[str, float],
    strategy_version: str,
    strategy_hash: str,
    as_of: datetime,
    min_notional: float = 1.0,
) -> list[Order]:
    out: list[Order] = []
    for d in approved:
        pos = account.positions.get(d.symbol)
        cur_qty = pos.qty if pos else 0.0
        t = d.target_weight
        closing = cur_qty != 0 and (t == 0 or (t > 0) != (cur_qty > 0))
        if closing:
            side = "sell" if cur_qty > 0 else "buy"
            out.append(
                Order(
                    symbol=d.symbol, side=side, notional=None, qty=abs(cur_qty),
                    client_order_id=client_order_id(strategy_hash, d.symbol, as_of, side),
                    strategy_version=strategy_version, strategy_hash=strategy_hash,
                    reason=f"close: target {t:+.4f}",
                )
            )
            continue
        price = prices.get(d.symbol)
        cur_notional = cur_qty * price if (cur_qty and price) else (pos.market_value if pos else 0.0)
        delta = t * account.equity - cur_notional
        if abs(delta) < min_notional:
            continue
        side = "buy" if delta > 0 else "sell"
        out.append(
            Order(
                symbol=d.symbol, side=side, notional=round(abs(delta), 2), qty=None,
                client_order_id=client_order_id(strategy_hash, d.symbol, as_of, side),
                strategy_version=strategy_version, strategy_hash=strategy_hash,
                reason=f"target {t:+.4f} from {cur_notional / account.equity:+.4f}",
            )
        )
    return out


def submit_orders(broker, orders: Sequence[Order], conn: sqlite3.Connection, dry_run: bool) -> list[tuple[str, str]]:
    """Never submits a client_order_id twice: skips ids already open at the broker or already in
    the orders table. The row is written BEFORE the broker call, so a crash mid-submit can at
    worst miss an order, never duplicate one. Dry run (argument or broker.dry_run) never calls
    broker.submit. Returns (client_order_id, status) per order."""
    dry = bool(dry_run or getattr(broker, "dry_run", False))
    open_ids = {o.client_order_id for o in broker.open_orders()}
    results: list[tuple[str, str]] = []
    for o in orders:
        cid = o.client_order_id
        if cid in open_ids or conn.execute("SELECT 1 FROM orders WHERE client_order_id = ?", (cid,)).fetchone():
            results.append((cid, "skipped_duplicate"))
            continue
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO orders (client_order_id, broker_order_id, symbol, side, notional, qty, type, time_in_force, "
            "status, submitted_at, dry_run, reason, strategy_version, strategy_hash) "
            "VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (cid, o.symbol, o.side, o.notional, o.qty, o.type, o.time_in_force,
             "dry_run" if dry else "submitting", now, int(dry), o.reason, o.strategy_version, o.strategy_hash),
        )
        conn.commit()
        if dry:
            log.info("dry run order %s %s %s notional=%s qty=%s", cid, o.side, o.symbol, o.notional, o.qty)
            results.append((cid, "dry_run"))
            continue
        try:
            bid = broker.submit(o)
        except Exception as e:  # recorded, never retried automatically
            conn.execute("UPDATE orders SET status = ?, reason = ? WHERE client_order_id = ?",
                         ("error", f"{o.reason} | {type(e).__name__}: {e}", cid))
            conn.commit()
            log.error("order %s failed: %s", cid, e)
            results.append((cid, "error"))
            continue
        conn.execute("UPDATE orders SET status = 'submitted', broker_order_id = ? WHERE client_order_id = ?", (bid, cid))
        conn.commit()
        results.append((cid, "submitted"))
    return results
