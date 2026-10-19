"""reconcile(): the ONLY constructor of AccountState.

Everything comes from the broker: positions, equity, cash, equity history. peak_equity is the max
of the broker-reported history and current equity; day_start_equity is the broker's previous
close equity (last_equity), falling back to the last history point before today, then current
equity. opened_at on each position comes from our fills log (the broker does not report it).
"""
from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from typing import Mapping, Optional

from sentry.execution.tradelog import open_since
from sentry.types import AccountState


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat()


def reconcile(
    broker,
    conn: sqlite3.Connection,
    strategy_version: str,
    now: datetime,
    expected_positions: Optional[Mapping[str, float]] = None,
) -> tuple[AccountState, list[dict]]:
    acct = broker.account()
    since = open_since(conn)
    positions = {s: replace(p, opened_at=since.get(s)) for s, p in broker.positions().items() if p.qty != 0}

    hist = tuple(sorted(acct.equity_history))
    peak = max([acct.equity, *(e for _, e in hist)])
    if acct.last_equity:
        day_start = acct.last_equity
    else:
        prior = [e for ts, e in hist if ts.date() < now.date()]
        day_start = prior[-1] if prior else acct.equity

    ts = _iso(now)
    conn.executemany(
        "INSERT OR REPLACE INTO positions_snapshot (ts, symbol, qty, avg_entry_price, market_value, equity, "
        "strategy_version) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(ts, p.symbol, p.qty, p.avg_entry_price, p.market_value, acct.equity, strategy_version) for p in positions.values()],
    )
    conn.execute(
        "INSERT OR REPLACE INTO equity_history (ts, equity, strategy_version) VALUES (?, ?, ?)",
        (ts, acct.equity, strategy_version),
    )
    conn.commit()

    diffs: list[dict] = []
    if expected_positions is not None:
        for sym in sorted(set(expected_positions) | set(positions)):
            exp = expected_positions.get(sym, 0.0)
            act = positions[sym].qty if sym in positions else 0.0
            if abs(exp - act) > 1e-9:
                diffs.append({"symbol": sym, "expected": exp, "actual": act, "reconcile_diff": act - exp})

    state = AccountState(
        as_of=now, equity=acct.equity, cash=acct.cash, positions=positions, equity_history=hist,
        day_start_equity=day_start, peak_equity=peak, is_paper=bool(acct.is_paper and getattr(broker, "is_paper", False)),
    )
    return state, diffs
