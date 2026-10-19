"""Read-only dashboard. The only state it changes is the login session.

Every metric is scoped to one strategy_version; there is deliberately no all-versions equity curve.
"""
from __future__ import annotations

import hmac
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from sentry.data.store import connect, default_db_path
from sentry.execution.tradelog import clock_start, trading_days
from sentry.web.alerts import STALE_AFTER, _detail, last_snapshot_as_of, latest_cycle_rows

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
COST_CEILING_USD = 10.0
# PLAN section 2.5 pre-registered numbers.
CHANGE_FLOOR_TRADES, CHANGE_FLOOR_DAYS, REVIEW_TRADES = 100, 60, 300
PAGES = [("/", "Overview"), ("/positions", "Positions"), ("/decisions", "Decisions"), ("/agents", "Agent views"),
         ("/blocks", "TA / FA"), ("/equity", "Equity"), ("/trades", "Trades"), ("/ic", "Forward IC"), ("/gates", "Gates"),
         ("/cycles", "Cycles"), ("/costs", "Cost")]

CSP = ("default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' data:; "
       "form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
SECURITY_HEADERS = {"Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff",
                    "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY", "Cache-Control": "no-store"}


def _num(v, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


TEMPLATES.env.filters["num"] = _num


def _json(raw) -> dict:
    return _detail(raw)


def _versions(conn) -> list[str]:
    rows = conn.execute(
        "SELECT strategy_version v, MAX(t) m FROM ("
        " SELECT strategy_version, ts t FROM equity_history"
        " UNION ALL SELECT strategy_version, as_of FROM features"
        " UNION ALL SELECT strategy_version, date FROM shadow_equity"
        " UNION ALL SELECT strategy_version, closed_at FROM closed_trades) GROUP BY strategy_version ORDER BY m DESC"
    ).fetchall()
    return [r["v"] for r in rows]


def _current_version(conn, requested: str | None) -> tuple[str | None, list[str]]:
    versions = _versions(conn)
    if requested and requested in versions:
        return requested, versions
    row = conn.execute("SELECT strategy_version FROM equity_history ORDER BY ts DESC LIMIT 1").fetchone()
    if row is None:
        row = conn.execute("SELECT strategy_version FROM features ORDER BY as_of DESC LIMIT 1").fetchone()
    return (row["strategy_version"] if row else None), versions


def _halts(conn) -> list[str]:
    out: list[str] = []
    for r in latest_cycle_rows(conn):
        if r["status"] == "halt":
            reasons = _json(r["detail"]).get("reasons") or []
            out.append(f"{r['step']}: " + (", ".join(map(str, reasons)) or "halt"))
    return out


def _cost_today(conn) -> float | None:
    row = conn.execute("SELECT usd FROM cost_daily WHERE date = ?", (datetime.now(timezone.utc).date().isoformat(),)).fetchone()
    return row["usd"] if row else None


def _reconcile_ok(conn) -> bool | None:
    rows = [r for r in latest_cycle_rows(conn) if r["step"] == "reconcile"]
    return None if not rows else all(r["status"] != "diff" for r in rows)


def _status(conn, now: datetime | None = None) -> dict:
    """Health strip inputs: snapshot age (stale after 18 h, as the alerts do) and today's cost against the ceiling."""
    now = now or datetime.now(timezone.utc)
    as_of, age_h = last_snapshot_as_of(conn), None
    if as_of:
        try:
            age_h = (now - datetime.fromisoformat(as_of)).total_seconds() / 3600
        except (TypeError, ValueError):
            age_h = None
    cost = _cost_today(conn)
    return {"as_of": as_of, "age_h": age_h, "stale": age_h is None or age_h > STALE_AFTER.total_seconds() / 3600,
            "cost": cost, "ceiling": COST_CEILING_USD, "over": cost is not None and cost > COST_CEILING_USD}


def _chart(series: dict[str, list[tuple[str, float]]], w: int = 640, h: int = 220) -> dict:
    """Inline-SVG polylines, each series indexed to its first value = 100 (baselines start at their own capital)."""
    dates = sorted({d for pts in series.values() for d, _ in pts})
    lines, lo, hi = [], float("inf"), float("-inf")
    idx = {d: i for i, d in enumerate(dates)}
    norm = {}
    for name, pts in series.items():
        if pts and pts[0][1]:
            norm[name] = [(d, v / pts[0][1] * 100) for d, v in pts]
            vals = [v for _, v in norm[name]]
            lo, hi = min(lo, *vals), max(hi, *vals)
    if not norm:
        return {"w": w, "h": h, "lines": [], "dates": dates, "lo": None, "hi": None}
    span = (hi - lo) or 1.0
    for i, (name, pts) in enumerate(norm.items()):
        xs = [8 + (w - 16) * (idx[d] / max(len(dates) - 1, 1)) for d, _ in pts]
        ys = [h - 8 - (h - 16) * ((v - lo) / span) for _, v in pts]
        lines.append({"name": name, "cls": f"s{i % 4}", "first": pts[0][0], "last": pts[-1][0], "end": norm[name][-1][1],
                      "n": len(pts), "points": " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))})
    return {"w": w, "h": h, "lines": lines, "dates": dates, "lo": lo, "hi": hi}


def create_app(db_path: str | None = None, testing_password: str | None = None,
               testing_secret: str | None = None) -> FastAPI:
    testing = testing_password is not None or testing_secret is not None
    password = testing_password if testing_password is not None else os.environ.get("SENTRY_DASH_PASSWORD")
    secret = testing_secret if testing_secret is not None else os.environ.get("SENTRY_SESSION_SECRET")
    if not password or not secret:
        raise RuntimeError("SENTRY_DASH_PASSWORD and SENTRY_SESSION_SECRET must both be set; refusing to start")
    path = str(db_path if db_path is not None else default_db_path())
    connect(path).close()  # make sure the schema exists; idempotent, the engine owns all later writes

    app = FastAPI(title="SENTRY", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        if request.url.path in ("/login", "/health") or request.session.get("ok"):
            return await call_next(request)
        return RedirectResponse("/login", status_code=303)

    # Added after the login middleware so it wraps it: the session is loaded before the check runs.
    app.add_middleware(SessionMiddleware, secret_key=secret, same_site="lax", max_age=14 * 86400,
                       https_only=(os.environ.get("SENTRY_ENV") == "prod" and not testing))

    @app.middleware("http")
    async def security_headers(request: Request, call_next):  # added last, so it wraps every response
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers[k] = v
        return response

    def ro() -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def render(request: Request, name: str, build):
        conn = ro()
        try:
            cur, versions = _current_version(conn, request.query_params.get("v"))
            ctx = {"version": cur, "versions": versions, "pages": PAGES, "path": request.url.path}
            ctx.update(build(conn, cur))
            return TEMPLATES.TemplateResponse(request, name, ctx)
        finally:
            conn.close()

    @app.get("/login")
    def login_form(request: Request):
        return TEMPLATES.TemplateResponse(request, "login.html", {"error": False})

    @app.post("/login")
    def login(request: Request, password_in: str = Form("", alias="password")):
        if hmac.compare_digest(password_in.encode(), password.encode()):
            request.session["ok"] = True
            return RedirectResponse("/", status_code=303)
        return TEMPLATES.TemplateResponse(request, "login.html", {"error": True}, status_code=401)

    @app.post("/logout")
    def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    @app.get("/health")
    def health():
        conn = ro()
        try:
            cur, _ = _current_version(conn, None)
            last = conn.execute("SELECT MAX(cycle_ts) m FROM health").fetchone()["m"]
            return JSONResponse({"last_cycle": last, "last_snapshot_as_of": last_snapshot_as_of(conn),
                                 "halts": _halts(conn), "cost_today_usd": _cost_today(conn),
                                 "reconcile_ok": _reconcile_ok(conn), "strategy_version": cur})
        finally:
            conn.close()

    @app.get("/")
    def overview(request: Request):
        def build(conn, v):
            if v is None:
                return {"empty": True}
            trades = conn.execute("SELECT COUNT(*) n FROM closed_trades WHERE strategy_version=?", (v,)).fetchone()["n"]
            days, start = trading_days(conn, v), clock_start(conn, v)  # dry-run burn-in days never count
            eq = conn.execute("SELECT ts, equity FROM equity_history WHERE strategy_version=? ORDER BY ts DESC LIMIT 1",
                              (v,)).fetchone()
            n_orders, n_dry = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(dry_run = 1), 0) FROM orders WHERE strategy_version=?", (v,)).fetchone()
            dry_run = start is None or (n_orders > 0 and n_orders == n_dry)
            return {"empty": False, "status": _status(conn), "dry_run": dry_run, "trades": trades, "days": days, "start": start, "equity": eq,
                    "halts": _halts(conn), "cost_today": _cost_today(conn), "reconcile_ok": _reconcile_ok(conn),
                    "floor_trades": CHANGE_FLOOR_TRADES, "floor_days": CHANGE_FLOOR_DAYS, "review": REVIEW_TRADES,
                    "floor_met": trades >= CHANGE_FLOOR_TRADES and days >= CHANGE_FLOOR_DAYS,
                    "review_met": trades >= REVIEW_TRADES}
        return render(request, "overview.html", build)

    @app.get("/positions")
    def positions(request: Request):
        def build(conn, v):
            ts = conn.execute("SELECT MAX(ts) m FROM positions_snapshot").fetchone()["m"]
            rows = conn.execute("SELECT * FROM positions_snapshot WHERE ts=? AND strategy_version=? ORDER BY symbol",
                                (ts, v)).fetchall() if ts else []
            return {"ts": ts, "rows": rows}
        return render(request, "positions.html", build)

    @app.get("/decisions")
    def decisions(request: Request, day: str | None = None):
        def build(conn, v):
            days = [r["d"] for r in conn.execute(
                "SELECT DISTINCT substr(as_of,1,10) d FROM decisions WHERE strategy_version=? ORDER BY d DESC LIMIT 60",
                (v,))]
            sel = day if day in days else (days[0] if days else None)
            rows = []
            for r in conn.execute(
                "SELECT d.*, f.row_json FROM decisions d LEFT JOIN features f "
                "ON f.symbol=d.symbol AND f.as_of=d.as_of AND f.strategy_hash=d.strategy_hash "
                "WHERE d.strategy_version=? AND substr(d.as_of,1,10)=? ORDER BY d.symbol", (v, sel)):
                rows.append({"r": r, "f": _json(r["row_json"]), "reason": _json(r["reason_json"])})
            return {"days": days, "day": sel, "rows": rows}
        return render(request, "decisions.html", build)

    @app.get("/agents")
    def agents(request: Request, symbol: str | None = None):
        def build(conn, v):
            q = ("SELECT a.* FROM agent_calls a WHERE EXISTS (SELECT 1 FROM features f WHERE f.symbol=a.symbol "
                 "AND f.as_of=a.as_of AND f.strategy_version=?)")
            args: list = [v]
            if symbol:
                q += " AND a.symbol=?"
                args.append(symbol)
            rows = [{"r": r, "v": _json(r["view_json"])}
                    for r in conn.execute(q + " ORDER BY a.as_of DESC, a.symbol LIMIT 50", args)]
            return {"rows": rows, "symbol": symbol or ""}
        return render(request, "agents.html", build)

    @app.get("/blocks")
    def blocks(request: Request):
        def build(conn, v):
            latest: dict[str, dict] = {}
            for r in conn.execute("SELECT symbol, as_of, row_json FROM features WHERE strategy_version=? "
                                  "ORDER BY as_of ASC", (v,)):
                latest[r["symbol"]] = {"as_of": r["as_of"], "f": _json(r["row_json"])}
            return {"items": sorted(latest.items())}
        return render(request, "blocks.html", build)

    @app.get("/equity")
    def equity(request: Request):
        def build(conn, v):
            real: dict[str, float] = {}
            for r in conn.execute("SELECT ts, equity FROM equity_history WHERE strategy_version=? ORDER BY ts", (v,)):
                real[r["ts"][:10]] = r["equity"]
            series = {"broker equity": sorted(real.items())}
            for r in conn.execute("SELECT baseline, date, equity FROM shadow_equity WHERE strategy_version=? "
                                  "ORDER BY baseline, date", (v,)):
                series.setdefault("shadow " + r["baseline"], []).append((r["date"], r["equity"]))
            return {"chart": _chart({k: s for k, s in series.items() if s})}
        return render(request, "equity.html", build)

    @app.get("/trades")
    def trades(request: Request):
        def build(conn, v):
            def q(sql):
                return conn.execute(sql, (v,)).fetchall()
            return {"orders": q("SELECT * FROM orders WHERE strategy_version=? ORDER BY submitted_at DESC LIMIT 100"),
                    "fills": q("SELECT * FROM fills WHERE strategy_version=? ORDER BY filled_at DESC LIMIT 100"),
                    "closed": q("SELECT * FROM closed_trades WHERE strategy_version=? ORDER BY closed_at DESC LIMIT 100")}
        return render(request, "trades.html", build)

    @app.get("/ic")
    def ic(request: Request):
        def build(conn, v):
            day = conn.execute("SELECT MAX(date) m FROM ic_daily WHERE strategy_version=?", (v,)).fetchone()["m"]
            rows = conn.execute("SELECT signal, horizon, ic, n FROM ic_daily WHERE strategy_version=? AND date=?",
                                (v, day)).fetchall() if day else []
            horizons = sorted({r["horizon"] for r in rows})
            cells = {(r["signal"], r["horizon"]): r for r in rows}
            return {"day": day, "horizons": horizons, "signals": sorted({r["signal"] for r in rows}), "cells": cells,
                    "need_days": CHANGE_FLOOR_DAYS, "need_trades": REVIEW_TRADES}
        return render(request, "ic.html", build)

    @app.get("/gates")
    def gates(request: Request):
        def build(conn, v):
            counts: dict[str, int] = {}
            for r in conn.execute("SELECT row_json FROM features WHERE strategy_version=?", (v,)):
                k = str(_json(r["row_json"]).get("gate_reason", "n/a"))
                counts[k] = counts.get(k, 0) + 1
            return {"counts": sorted(counts.items(), key=lambda kv: -kv[1])}
        return render(request, "gates.html", build)

    @app.get("/cycles")
    def cycles(request: Request):
        def build(conn, v):
            return {"rows": conn.execute("SELECT * FROM health ORDER BY cycle_ts DESC, step LIMIT 200").fetchall()}
        return render(request, "cycles.html", build)

    @app.get("/costs")
    def costs(request: Request):
        def build(conn, v):
            return {"rows": conn.execute("SELECT * FROM cost_daily ORDER BY date DESC LIMIT 90").fetchall()}
        return render(request, "costs.html", build)

    return app
