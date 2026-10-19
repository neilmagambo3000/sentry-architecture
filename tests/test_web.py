import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from sentry.data.store import connect
from sentry.web.alerts import check_alerts, send_alert
from sentry.web.app import create_app

PAGES = ["/", "/positions", "/decisions", "/agents", "/blocks", "/equity", "/trades", "/ic", "/gates", "/cycles", "/costs"]
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def _client(path, logged_in=True):
    c = TestClient(create_app(str(path), testing_password="pw", testing_secret="s3cret"), follow_redirects=False)
    if logged_in:
        assert c.post("/login", data={"password": "pw"}).status_code == 303
    return c


def _seed(path):
    conn = connect(str(path))
    for v, eq, day in (("v1.0.0", 100.0, "2026-09-29"), ("v1.1.0", 200.0, "2026-09-30")):
        row = json.dumps({"gate_reason": f"gate_{v}",
                          "disc_direction": 1, "disc_confidence": 0.7, "disagreement": 0.2})
        conn.execute("INSERT INTO features VALUES (?,?,?,?,?)", ("AAA", day + "T12:00:00+00:00", "h" + v, v, row))
        conn.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?)",
                     ("AAA", day + "T12:00:00+00:00", 1, 0.05, 9.0, 0.3, "{}", "p", v, "h" + v))
        conn.execute("INSERT INTO equity_history VALUES (?,?,?)", (day + "T21:00:00+00:00", eq, v))
        conn.execute("INSERT INTO shadow_equity VALUES (?,?,?,?,?,?)", (day, "ta_only", v, eq * 2, 0.0, 0.0))
        conn.execute("INSERT INTO closed_trades VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     ("AAA", day + "a", day + "b", 1, 1.0, 10.0, 11.0, 1.0, "stop", v, "h" + v))
        conn.execute("INSERT INTO positions_snapshot VALUES (?,?,?,?,?,?,?)",
                     (day + "T21:00:00+00:00", "AAA", 1.0, 10.0, 11.0, eq, v))
        view = json.dumps({"direction": 1, "confidence": 0.7, "horizon_days": 5, "rationale": "because",
                           "evidence_ids": ["a1"]})
        conn.execute("INSERT INTO agent_calls (agent,symbol,as_of,prompt_hash,ablation_key,cost_usd,view_json)"
                     " VALUES ('disc','AAA',?,?,'k',0.01,?)", (day + "T12:00:00+00:00", "p" + v, view))
    # SYNTHETIC rows for the trades and forward IC pages
    conn.execute("INSERT INTO orders VALUES ('c1','b1','AAA','buy',50.0,NULL,'limit','day','filled','2026-09-30T14:00',1,'r','v1.1.0','h')")
    conn.execute("INSERT INTO orders VALUES ('c2','b2','BBB','sell',60.0,NULL,'limit','day','new','2026-09-30T14:01',0,'r','v1.1.0','h')")
    conn.execute("INSERT INTO fills VALUES ('b2','c2','BBB','sell',1.0,12.0,'2026-09-30T14:02',12.1,'v1.1.0','h')")
    conn.execute("INSERT INTO ic_daily VALUES ('2026-09-29','v1.1.0',5,'old_sig',0.9,10)")
    conn.execute("INSERT INTO ic_daily VALUES ('2026-09-30','v1.1.0',5,'sig_a',0.123,40)")
    conn.execute("INSERT INTO ic_daily VALUES ('2026-09-30','v1.1.0',10,'sig_a',-0.05,38)")
    conn.execute("INSERT INTO ic_daily VALUES ('2026-09-30','v1.0.0',5,'other_ver',0.5,9)")
    conn.execute("INSERT INTO cost_daily VALUES (?,?)", (NOW.date().isoformat(), 1.5))
    conn.execute("INSERT INTO health VALUES ('2026-09-30T12:00','snapshot','ok',?)",
                 (json.dumps({"as_of": "2026-09-30T12:00:00+00:00"}),))
    conn.execute("INSERT INTO health VALUES ('2026-09-30T12:00','risk','halt',?)", (json.dumps({"reasons": ["dd"]}),))
    conn.execute("INSERT INTO health VALUES ('2026-09-30T12:00','reconcile','diff','{}')")
    conn.commit()
    conn.close()


def test_login_required_and_wrong_password(tmp_path):
    c = _client(tmp_path / "t.db", logged_in=False)
    for p in PAGES:
        r = c.get(p)
        assert r.status_code == 303 and r.headers["location"] == "/login"
    assert c.get("/login").status_code == 200
    assert c.post("/login", data={"password": "nope"}).status_code == 401
    assert c.get("/").status_code == 303


def test_refuses_to_start_without_secrets(tmp_path, monkeypatch):
    monkeypatch.delenv("SENTRY_DASH_PASSWORD", raising=False)
    monkeypatch.delenv("SENTRY_SESSION_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="SENTRY_DASH_PASSWORD"):
        create_app(str(tmp_path / "t.db"))


def test_health_empty_db_is_public(tmp_path):
    c = _client(tmp_path / "t.db", logged_in=False)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"last_cycle": None, "last_snapshot_as_of": None, "halts": [], "cost_today_usd": None,
                        "reconcile_ok": None, "strategy_version": None}


def test_pages_render_empty(tmp_path):
    c = _client(tmp_path / "t.db")
    for p in PAGES:
        r = c.get(p)
        assert r.status_code == 200, p
        assert "no data yet" in r.text.lower(), p


def test_pages_render_seeded_and_version_filter(tmp_path):
    p = tmp_path / "t.db"
    _seed(p)
    c = _client(p)
    for page in PAGES:
        assert c.get(page).status_code == 200, page
    assert c.get("/health").json()["halts"] == ["risk: dd"]
    assert c.get("/health").json()["reconcile_ok"] is False
    # default is the latest version; the selector switches to the other and never mixes them
    assert "gate_v1.1.0" in c.get("/gates").text and "gate_v1.0.0" not in c.get("/gates").text
    old = c.get("/gates?v=v1.0.0").text
    assert "gate_v1.0.0" in old and "gate_v1.1.0" not in old
    assert "200.00" in c.get("/").text and "100.00" in c.get("/?v=v1.0.0").text
    assert "<polyline" in c.get("/equity").text
    assert "because" in c.get("/agents").text
    assert "health only until the floor is met" in c.get("/").text.lower()


def test_no_write_routes():
    app = create_app(":memory:", testing_password="pw", testing_secret="s")
    bad = [(r.path, m) for r in app.routes for m in getattr(r, "methods", ()) or ()
           if m not in ("GET", "HEAD") and r.path not in ("/login", "/logout")]
    assert bad == []


def test_check_alerts():
    conn = connect(":memory:")
    assert check_alerts(conn, NOW, 10.0) == []
    conn.execute("INSERT INTO health VALUES ('c1','snapshot','ok',?)",
                 (json.dumps({"as_of": (NOW - timedelta(hours=19)).isoformat()}),))
    conn.execute("INSERT INTO health VALUES ('c1','risk','halt',?)", (json.dumps({"reasons": ["total_dd"]}),))
    conn.execute("INSERT INTO health VALUES ('c1','reconcile','diff','{}')")
    conn.execute("INSERT INTO cost_daily VALUES (?,10.0)", (NOW.date().isoformat(),))
    for i in range(10):
        conn.execute("INSERT INTO agent_calls (agent,symbol,as_of,prompt_hash,ablation_key,error) VALUES "
                     "('disc',?,?,'p','k',?)", (f"S{i}", NOW.isoformat(), "boom" if i < 2 else None))
    msgs = " | ".join(check_alerts(conn, NOW, 10.0))
    for needle in ("data stale", "agent error rate", "ceiling", "total_dd", "reconciliation diff"):
        assert needle in msgs
    # exactly 10% is not above the threshold; fresh snapshot, cost below ceiling
    conn.execute("UPDATE agent_calls SET error=NULL WHERE symbol='S1'")
    conn.execute("UPDATE cost_daily SET usd=1.0")
    conn.execute("UPDATE health SET detail=? WHERE step='snapshot'", (json.dumps({"as_of": NOW.isoformat()}),))
    msgs = " | ".join(check_alerts(conn, NOW, 10.0))
    assert "stale" not in msgs and "error rate" not in msgs and "ceiling" not in msgs


def test_send_alert_noop_without_env(monkeypatch):
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    assert send_alert("x") is None


def test_new_pages_content_and_scope(tmp_path):
    p = tmp_path / "t.db"
    _seed(p)
    c = _client(p)
    t = c.get("/trades").text
    assert "DRY RUN" in t and "BBB" in t and "paper" in t
    ic = c.get("/ic").text
    assert "sig_a" in ic and "0.123" in ic and "old_sig" not in ic and "other_ver" not in ic
    assert "60 decision dates" in ic and "300 closed trades" in ic
    assert "other_ver" in c.get("/ic?v=v1.0.0").text
    assert "DRY RUN" in c.get("/").text or "PAPER CLOCK" in c.get("/").text


def test_overview_mode_dry_run_vs_clock(tmp_path):
    p = tmp_path / "t.db"
    _seed(p)
    conn = connect(str(p))
    conn.execute("DELETE FROM orders WHERE dry_run=0")
    conn.commit()
    conn.close()
    assert "DRY RUN" in _client(p).get("/").text
    conn = connect(str(p))
    conn.execute("INSERT INTO orders VALUES ('c3','b3','CCC','buy',1.0,NULL,'limit','day','new','2026-09-30T15:00',0,'r','v1.1.0','h')")
    conn.commit()
    conn.close()
    assert "PAPER CLOCK" in _client(p).get("/").text


def test_security_headers_and_no_external_requests(tmp_path):
    p = tmp_path / "t.db"
    _seed(p)
    c = _client(p, logged_in=False)
    for r in (c.get("/"), c.get("/login"), c.get("/health")):
        h = r.headers
        assert "default-src 'self'" in h["content-security-policy"] and "frame-ancestors 'none'" in h["content-security-policy"]
        assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY"
        assert h["referrer-policy"] == "no-referrer" and h["cache-control"] == "no-store"
    c = _client(p)
    for page in PAGES + ["/login"]:
        html = c.get(page).text.lower()
        assert "unpkg" not in html and "htmx" not in html and "hx-" not in html and "onchange" not in html, page
        assert "<script" not in html and "http://" not in html and "https://" not in html, page


def test_every_route_opens_db_read_only(tmp_path, monkeypatch):
    import sqlite3

    p = tmp_path / "t.db"
    _seed(p)
    c = _client(p)
    opened = []
    real = sqlite3.connect

    def spy(target, *a, **k):
        opened.append(str(target))
        return real(target, *a, **k)

    monkeypatch.setattr("sentry.web.app.sqlite3.connect", spy)
    for page in PAGES + ["/health"]:
        assert c.get(page).status_code == 200
    assert len(opened) == len(PAGES) + 1 and all(o.endswith("?mode=ro") for o in opened)
    with pytest.raises(sqlite3.OperationalError):
        real(f"file:{p}?mode=ro", uri=True).execute("INSERT INTO cost_daily VALUES ('x', 1)")
