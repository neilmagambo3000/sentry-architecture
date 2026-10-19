"""Fills and closed trades, stamped with strategy_version / strategy_hash.

Fills come from the broker (Broker.fills), never assumed. closed_trades is rebuilt from the full
fills table on every record (average-cost method): a position opened and then fully closed is one
closed trade; a fill that goes through zero closes the trade and opens a new one with the
remainder at that fill's price. A closed trade is stamped with the version/hash of its closing fill.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Iterable, Mapping, Optional

from sentry.execution.broker import Fill

_EPS = 1e-9


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat()


def record_fills(
    conn: sqlite3.Connection,
    fills: Iterable[Fill],
    strategy_version: str,
    strategy_hash: str,
    ref_prices: Optional[Mapping[str, float]] = None,
) -> int:
    """Insert new fills (idempotent on (broker_order_id, filled_at)), then rebuild closed_trades.
    ref_prices (symbol -> reference price, e.g. the 09:35 mid) is stored for realised slippage."""
    n = 0
    for f in fills:
        cur = conn.execute(
            "INSERT OR IGNORE INTO fills (broker_order_id, client_order_id, symbol, side, qty, price, filled_at, "
            "ref_price, strategy_version, strategy_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f.broker_order_id, f.client_order_id, f.symbol, f.side, f.qty, f.price, _iso(f.filled_at),
             (ref_prices or {}).get(f.symbol), strategy_version, strategy_hash),
        )
        n += cur.rowcount
    rebuild_closed_trades(conn)
    return n


def _walk(conn: sqlite3.Connection) -> tuple[list[tuple], dict[str, dict]]:
    closed: list[tuple] = []
    state: dict[str, dict] = {}
    rows = conn.execute(
        "SELECT f.symbol, f.side, f.qty, f.price, f.filled_at, f.strategy_version, f.strategy_hash, o.reason "
        "FROM fills f LEFT JOIN orders o ON o.client_order_id = f.client_order_id ORDER BY f.filled_at, f.rowid"
    ).fetchall()
    for sym, side, qty, price, ts, ver, h, reason in rows:
        dq = qty if side == "buy" else -qty
        s = state.get(sym)
        if s is None or abs(s["q"]) < _EPS:
            state[sym] = {"q": dq, "avg": price, "opened_at": ts, "pnl": 0.0, "closed_q": 0.0, "exit_val": 0.0}
            continue
        sgn = 1 if s["q"] > 0 else -1
        if (dq > 0) == (s["q"] > 0):
            s["avg"] = (s["q"] * s["avg"] + dq * price) / (s["q"] + dq)
            s["q"] += dq
            continue
        c = min(abs(dq), abs(s["q"]))
        s["pnl"] += sgn * c * (price - s["avg"])
        s["closed_q"] += c
        s["exit_val"] += c * price
        s["q"] -= sgn * c
        if abs(s["q"]) < _EPS:
            closed.append((sym, s["opened_at"], ts, sgn, s["closed_q"], s["avg"], s["exit_val"] / s["closed_q"],
                           s["pnl"], reason, ver, h))
            rest = abs(dq) - c
            if rest > _EPS:
                state[sym] = {"q": -sgn * rest, "avg": price, "opened_at": ts, "pnl": 0.0, "closed_q": 0.0, "exit_val": 0.0}
            else:
                del state[sym]
    return closed, state


def rebuild_closed_trades(conn: sqlite3.Connection) -> None:
    closed, _ = _walk(conn)
    conn.execute("DELETE FROM closed_trades")
    conn.executemany(
        "INSERT INTO closed_trades (symbol, opened_at, closed_at, side, qty, entry_price, exit_price, pnl, exit_reason, "
        "strategy_version, strategy_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        closed,
    )
    conn.commit()


def open_since(conn: sqlite3.Connection) -> dict[str, datetime]:
    """symbol -> when the currently open position was opened, per the fills log."""
    _, state = _walk(conn)
    return {s: datetime.fromisoformat(v["opened_at"]) for s, v in state.items() if abs(v["q"]) >= _EPS}


def closed_trades_count(conn: sqlite3.Connection, strategy_version: str) -> int:
    return conn.execute("SELECT COUNT(*) FROM closed_trades WHERE strategy_version = ?", (strategy_version,)).fetchone()[0]


def clock_start(conn: sqlite3.Connection, strategy_version: str) -> Optional[str]:
    """Date (YYYY-MM-DD, UTC) the per-version clock starts for this version: the first non-dry-run order
    accepted by the broker, or the first broker fill if earlier. Dry-run burn-in days never count.
    None until then."""
    row = conn.execute(
        "SELECT MIN(d) FROM ("
        " SELECT substr(submitted_at, 1, 10) d FROM orders"
        "  WHERE strategy_version = ? AND dry_run = 0 AND broker_order_id IS NOT NULL"
        " UNION ALL SELECT substr(filled_at, 1, 10) FROM fills WHERE strategy_version = ?)",
        (strategy_version, strategy_version),
    ).fetchone()
    return row[0] if row else None


def trading_days(conn: sqlite3.Connection, strategy_version: str) -> int:
    """Distinct dates with a reconcile under this version (equity_history is written every cycle),
    counted from clock_start(); 0 while the version has only run in dry run."""
    start = clock_start(conn, strategy_version)
    if start is None:
        return 0
    return conn.execute(
        "SELECT COUNT(DISTINCT substr(ts, 1, 10)) FROM equity_history WHERE strategy_version = ? AND substr(ts, 1, 10) >= ?",
        (strategy_version, start),
    ).fetchone()[0]
