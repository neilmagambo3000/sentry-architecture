"""Deprecation and cost-drift alerts. All data SYNTHETIC."""
from datetime import date, datetime, timezone

import pytest

from sentry.web.alerts import check_alerts, cost_per_decision_drift

NOW = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)


def with_dep(settings, d, mp):
    mp.setenv("SENTRY_MODEL_DEPRECATION_DATE", d.isoformat())
    return settings


def test_deprecation_alert(settings, db, monkeypatch):
    monkeypatch.delenv("SENTRY_MODEL_DEPRECATION_DATE", raising=False)
    assert check_alerts(db, NOW, 10.0, settings=settings) == []
    assert check_alerts(db, NOW, 10.0, settings=with_dep(settings, date(2026, 11, 7), monkeypatch)) == []  # 31 days away
    msgs = check_alerts(db, NOW, 10.0, settings=with_dep(settings, date(2026, 11, 6), monkeypatch))  # 30 days away
    assert len(msgs) == 1 and "deprecation 2026-11-06" in msgs[0] and "30 days" in msgs[0]
    assert "deprecation" in check_alerts(db, NOW, 10.0, settings=with_dep(settings, date(2026, 10, 1), monkeypatch))[0]

    monkeypatch.setenv("SENTRY_MODEL_DEPRECATION_DATE", "not-a-date")
    with pytest.raises(ValueError):
        check_alerts(db, NOW, 10.0, settings=settings)


def day(db, d, n, usd, dry):
    for i in range(n):
        db.execute("INSERT INTO decisions VALUES (?,?,1,0.1,NULL,0.5,'{}','p','v1','h')", (f"S{i}", f"{d}T12:00:00+00:00"))
    db.execute("INSERT INTO agent_calls (agent,symbol,as_of,prompt_hash,ablation_key,cost_usd) VALUES "
               "('disc','S0',?,'p','k',?)", (f"{d}T12:00:00+00:00", usd))
    if dry:
        db.execute("INSERT INTO orders (client_order_id,status,submitted_at,dry_run,strategy_version,strategy_hash) "
                   "VALUES (?, 'dry_run', ?, 1, 'v1', 'h')", (f"c{d}", f"{d}T12:01:00+00:00"))


def test_cost_drift_alert(settings, db):
    assert cost_per_decision_drift(db, "2026-10-07") is None
    for i in (1, 2):
        day(db, f"2026-09-0{i}", 10, 1.0, dry=True)  # burn-in: $0.10 per decision
    for i in range(1, 6):
        day(db, f"2026-10-0{i}", 10, 1.4, dry=False)  # $0.14: 1.4x, under the line
    assert cost_per_decision_drift(db, "2026-10-07")is not None
    b, c = cost_per_decision_drift(db, "2026-10-07")
    assert abs(b - 0.1) < 1e-9 and abs(c - 0.14) < 1e-9
    assert check_alerts(db, NOW, 10.0, settings=settings) == []
    for i in range(1, 6):
        db.execute("UPDATE agent_calls SET cost_usd=1.6 WHERE substr(as_of,1,10)=?", (f"2026-10-0{i}",))
    msgs = check_alerts(db, NOW, 10.0, settings=settings)
    assert len(msgs) == 1 and "1.5x the burn-in" in msgs[0]


def test_zero_burn_in_never_alerts(settings, db):
    day(db, "2026-09-01", 10, 0.0, dry=True)
    day(db, "2026-10-01", 10, 5.0, dry=False)
    assert check_alerts(db, NOW, 10.0, settings=settings) == []
