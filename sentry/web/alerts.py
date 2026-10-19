"""Alerts (PLAN Phase 2): a webhook post plus the checks that decide what to say."""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta

import httpx

STALE_AFTER = timedelta(hours=18)


def deprecation_date() -> date | None:
    """SENTRY_MODEL_DEPRECATION_DATE (YYYY-MM-DD): Anthropic's announced retirement date for the pinned model.
    An env var, not models.yaml, because models.yaml is hashed into strategy_hash and a notice arriving
    after the freeze must not stop a restart. An unparseable value raises: fail closed, the alert check
    then fails loudly instead of going silent."""
    raw = (os.environ.get("SENTRY_MODEL_DEPRECATION_DATE") or "").strip()
    return date.fromisoformat(raw) if raw else None


def send_alert(text: str) -> None:
    """POST {"text": ...} to ALERT_WEBHOOK_URL. No-op when unset; never raises."""
    url = os.environ.get("ALERT_WEBHOOK_URL")
    if not url:
        return
    try:
        httpx.post(url, json={"text": text, "content": text}, timeout=5.0)
    except Exception:  # noqa: BLE001 - an alert failure must never take the engine down
        pass


def _detail(raw: str | None) -> dict:
    try:
        d = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def latest_cycle_rows(conn) -> list:
    return conn.execute(
        "SELECT cycle_ts, step, status, detail FROM health "
        "WHERE cycle_ts = (SELECT MAX(cycle_ts) FROM health)"
    ).fetchall()


def last_snapshot_as_of(conn) -> str | None:
    row = conn.execute(
        "SELECT detail FROM health WHERE step='snapshot' AND status='ok' ORDER BY cycle_ts DESC LIMIT 1"
    ).fetchone()
    return _detail(row["detail"]).get("as_of") if row else None


COST_DRIFT_X, LIVE_DAYS = 1.5, 5


def cost_per_decision_drift(conn, today: str) -> tuple[float, float] | None:
    """(burn-in, current) USD per decision, or None when either side has no data.
    Per day: sum(agent_calls.cost_usd) / count(decisions), both keyed on substr(as_of, 1, 10).
    Burn-in = mean over days that have a dry-run order (orders.dry_run=1, day = substr(submitted_at, 1, 10));
    current = mean over the last 5 other decision days on or before today. Versions are not split."""
    dry = {r[0] for r in conn.execute("SELECT DISTINCT substr(submitted_at, 1, 10) FROM orders WHERE dry_run = 1")}
    per_day = conn.execute(
        "SELECT d.day, d.n, COALESCE(c.usd, 0) FROM "
        "(SELECT substr(as_of, 1, 10) AS day, COUNT(*) AS n FROM decisions GROUP BY 1) d "
        "LEFT JOIN (SELECT substr(as_of, 1, 10) AS day, SUM(cost_usd) AS usd FROM agent_calls GROUP BY 1) c "
        "ON c.day = d.day WHERE d.day <= ? ORDER BY d.day", (today,)).fetchall()
    burn = [usd / n for day, n, usd in per_day if day in dry]
    live = [usd / n for day, n, usd in per_day if day not in dry][-LIVE_DAYS:]
    if not burn or not live:
        return None
    return sum(burn) / len(burn), sum(live) / len(live)


def check_alerts(conn, now: datetime, cost_ceiling: float, agent_error_threshold: float = 0.10,
                 settings=None) -> list[str]:
    """Return one message per active condition. `now` must be timezone-aware.
    With `settings`, also checks the model deprecation date (alert from 30 days before) and cost per
    decision drift (see cost_per_decision_drift; alert when current > 1.5x a positive burn-in figure)."""
    out: list[str] = []
    today = now.date().isoformat()

    dep = deprecation_date()
    if settings and dep and now.date() >= dep - timedelta(days=30):
        out.append(f"model {settings.models.model_id} deprecation {dep.isoformat()} is "
                   f"{(dep - now.date()).days} days away: plan a version bump")
    if settings and (d := cost_per_decision_drift(conn, today)):
        if d[0] > 0 and d[1] > COST_DRIFT_X * d[0]:
            out.append(f"cost per decision ${d[1]:.4f} is above {COST_DRIFT_X}x the burn-in ${d[0]:.4f}")

    as_of = last_snapshot_as_of(conn)
    if as_of:
        try:
            age = now - datetime.fromisoformat(as_of)
            if age > STALE_AFTER:
                out.append(f"data stale: last snapshot {as_of} is {age.total_seconds() / 3600:.1f} h old")
        except (TypeError, ValueError):
            out.append(f"data stale: unreadable snapshot time {as_of!r}")

    r = conn.execute(
        "SELECT COUNT(*) AS n, SUM(CASE WHEN error IS NOT NULL AND error != '' THEN 1 ELSE 0 END) AS e "
        "FROM agent_calls WHERE substr(as_of, 1, 10) = ?",
        (today,),
    ).fetchone()
    if r["n"] and (r["e"] or 0) / r["n"] > agent_error_threshold:
        out.append(f"agent error rate {(r['e'] or 0)}/{r['n']} today is above {agent_error_threshold:.0%}")

    c = conn.execute("SELECT usd FROM cost_daily WHERE date = ?", (today,)).fetchone()
    if c and c["usd"] >= cost_ceiling:
        out.append(f"LLM cost today ${c['usd']:.2f} reached the ceiling ${cost_ceiling:.2f}")

    for row in latest_cycle_rows(conn):
        if row["step"] == "risk" and row["status"] == "halt":
            reasons = _detail(row["detail"]).get("reasons") or []
            out.append("halt: " + (", ".join(map(str, reasons)) or "unspecified"))
        elif row["step"] == "reconcile" and row["status"] == "diff":
            out.append(f"reconciliation diff in cycle {row['cycle_ts']}")
    return out
