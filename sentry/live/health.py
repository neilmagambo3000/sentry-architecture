"""Health and cost rows in the dashboard convention (sentry/web/app.py, sentry/web/alerts.py).

health(cycle_ts, step, status, detail): cycle_ts is the cycle's snapshot as_of (ISO UTC), so re-running
the same as_of overwrites its own rows instead of adding new ones. Steps: snapshot, features, layer_a,
risk, orders, reconcile, shadow, cycle. Statuses: ok, error, halt, diff, skipped. detail is JSON.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from typing import Optional

from sentry.data.writes import iso_utc

STEPS = ("snapshot", "features", "layer_a", "risk", "orders", "reconcile", "shadow", "cycle")
STATUSES = ("ok", "error", "halt", "diff", "skipped")


def write_health(conn: sqlite3.Connection, as_of: datetime, step: str, status: str, detail: Optional[dict] = None) -> None:
    if step not in STEPS or status not in STATUSES:
        raise ValueError(f"unknown health step/status {step!r}/{status!r}")
    conn.execute("INSERT OR REPLACE INTO health (cycle_ts, step, status, detail) VALUES (?, ?, ?, ?)",
                 (iso_utc(as_of), step, status, json.dumps(detail or {}, sort_keys=True, default=str)))
    conn.commit()


def read_health(conn: sqlite3.Connection, as_of: datetime, step: str) -> Optional[tuple[str, dict]]:
    row = conn.execute("SELECT status, detail FROM health WHERE cycle_ts = ? AND step = ?",
                       (iso_utc(as_of), step)).fetchone()
    return None if row is None else (row[0], json.loads(row[1] or "{}"))


def ensure_cost_row(conn: sqlite3.Connection, day: date) -> float:
    """Make sure cost_daily has a row for `day` (0.0 when nothing was spent); return the day's USD."""
    conn.execute("INSERT OR IGNORE INTO cost_daily (date, usd) VALUES (?, 0.0)", (day.isoformat(),))
    conn.commit()
    return float(conn.execute("SELECT usd FROM cost_daily WHERE date = ?", (day.isoformat(),)).fetchone()[0])
