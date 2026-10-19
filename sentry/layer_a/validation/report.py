"""Version-scoped paper report, rendered only on request.

Writes <SENTRY_DATA_DIR>/reports/<date>-<strategy_version>.md. It is a paper, forward-only report with no backtest;
the minimum-sample status is printed at the top. Generated reports are not committed.
Usage: .venv/bin/python -m sentry.layer_a.validation.report --version v1.0.0 --date 2026-10-01
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from sentry.config import ROOT, Settings, load_settings
from sentry.execution.tradelog import clock_start, closed_trades_count, trading_days
from sentry.features.version import tags
from sentry.layer_a.validation.ic import ic_table
from sentry.web.alerts import _detail

CHANGE_FLOOR_TRADES, CHANGE_FLOOR_DAYS, REVIEW_TRADES = 100, 60, 300  # pre-registered
MIN_RETURNS, MIN_IC_DATES, IC_T, INCR_T, COST_X = 20, 60, 2.0, 1.5, 1.5  # pre-registered review criteria
PASS, FAIL, NE = "PASS", "FAIL", "NOT EVALUABLE"


def sharpe(equity: pd.Series) -> float:
    """mean/std(ddof=1)*sqrt(252) of daily returns; nan (shown n/a) with < 20 returns or std 0."""
    r = equity.pct_change().dropna()
    if len(r) < MIN_RETURNS:
        return np.nan
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(252)) if sd > 1e-12 else np.nan


def max_drawdown(equity: pd.Series) -> float:
    return float((1 - equity / equity.cummax()).max()) if len(equity) else np.nan


def _series(conn: sqlite3.Connection, v: str, start: str | None) -> dict[str, pd.Series]:
    """Daily equity per series: real (last equity_history row per day) and each shadow baseline."""
    out = {}
    if start:
        rows = conn.execute("SELECT substr(ts, 1, 10), equity FROM equity_history WHERE strategy_version = ? "
                            "AND substr(ts, 1, 10) >= ? ORDER BY ts", (v, start)).fetchall()
        out["real"] = pd.Series({d: e for d, e in rows}, dtype=float)  # later rows win: last row per day
    for b in ("ta_only", "random"):
        rows = conn.execute("SELECT date, equity FROM shadow_equity WHERE strategy_version = ? AND baseline = ? "
                            "AND date >= ? ORDER BY date", (v, b, start or "")).fetchall()
        if rows:
            out[b] = pd.Series({d: e for d, e in rows}, dtype=float)
    return out


def _metrics(series: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame([[k, len(e) - 1, sharpe(e), e.iloc[-1] / e.iloc[0] - 1 if len(e) else np.nan, max_drawdown(e)]
                         for k, e in series.items()], columns=["series", "returns", "sharpe", "total_return", "max_drawdown"])


def _cost_per_decision(conn: sqlite3.Connection, v: str, where: str, args: tuple) -> float | None:
    """sum(cost_daily.usd) over the matching decision dates / decisions on them; None without decisions."""
    days = [r[0] for r in conn.execute(f"SELECT DISTINCT substr(as_of, 1, 10) FROM decisions WHERE strategy_version = ? {where}", (v, *args))]
    if not days:
        return None
    n = conn.execute(f"SELECT COUNT(*) FROM decisions WHERE strategy_version = ? {where}", (v, *args)).fetchone()[0]
    usd = conn.execute(f"SELECT COALESCE(SUM(usd), 0) FROM cost_daily WHERE date IN ({','.join('?' * len(days))})", days).fetchone()[0]
    return usd / n


def _dates(conn: sqlite3.Connection, v: str) -> set[str]:
    return {r[0] for r in conn.execute("SELECT DISTINCT substr(as_of, 1, 10) FROM decisions WHERE strategy_version = ?", (v,))}


def _health(conn: sqlite3.Connection, lo: str | None, hi: str | None, step: str | None = None) -> list[tuple]:
    """health rows with lo <= cycle date <= hi (None = open). health has no version column, so rows are
    selected by date range, not by version: cycles that stopped before any decision still count."""
    rows = conn.execute("SELECT cycle_ts, step, status, detail FROM health ORDER BY cycle_ts").fetchall()
    return [tuple(r) for r in rows if (lo is None or r[0][:10] >= lo) and (hi is None or r[0][:10] <= hi)
            and (step is None or r[1] == step)]


def _lo(conn: sqlite3.Connection, v: str, dates: set[str]) -> str | None:
    """Start of the health window: the clock start, else the first decision date."""
    return clock_start(conn, v) or (min(dates) if dates else None)


def evaluate_criteria(conn: sqlite3.Connection, v: str, ic: pd.DataFrame, feats: pd.DataFrame,
                      metrics: pd.DataFrame, day: str | None = None) -> list[tuple[str, str, str, str]]:
    """(id, criterion, status, numbers) for the pre-registered criteria i..v."""
    enough = closed_trades_count(conn, v) >= REVIEW_TRADES
    dates = _dates(conn, v)
    sh = dict(zip(metrics["series"], metrics["sharpe"], strict=True)) if len(metrics) else {}

    def grade(ok: bool | None) -> str:
        return NE if ok is None or not enough else PASS if ok else FAIL

    known = all(k in sh and pd.notna(sh[k]) for k in ("real", "ta_only", "random"))
    r1 = grade(sh["real"] > sh["ta_only"] and sh["real"] > sh["random"] if known else None)
    n1 = " ".join(f"{k}={_fmt(float(sh[k])) if k in sh else 'n/a'}" for k in ("real", "ta_only", "random"))

    hz = feats["disc_horizon"].dropna() if "disc_horizon" in feats else pd.Series(dtype=float)
    h = int(hz.mode().iloc[0]) if len(hz) else None
    row = ic[(ic["signal"] == "disc") & (ic["horizon"] == h)] if h is not None and len(ic) else ic.iloc[0:0]
    if len(row):
        x = row.iloc[0]
        ok2 = None if pd.isna(x.t_stat) or pd.isna(x.incr_t) else bool(x.n_dates >= MIN_IC_DATES and x.t_stat >= IC_T and x.incr_t >= INCR_T)
        n2 = f"horizon={h} t={_fmt(float(x.t_stat))} N={x.n_dates} incr_t={_fmt(float(x.incr_t))}"
    else:
        ok2, n2 = None, f"declared horizon={h} not in the IC table"

    lo = _lo(conn, v, dates)
    halts = [r for r in _health(conn, lo, day, "risk") if r[2] == "halt" and any(
        str(x).startswith("total drawdown") for x in (_detail(r[3]).get("reasons") or []))]
    cur = burn = None
    start = clock_start(conn, v)
    if start:  # live = on or after the clock start, burn-in = dry-run dates before it
        cur = _cost_per_decision(conn, v, "AND substr(as_of, 1, 10) >= ?", (start,))
        burn = _cost_per_decision(conn, v, "AND substr(as_of, 1, 10) < ?", (start,))
    ok4 = None if cur is None or not burn else cur <= COST_X * burn
    diffs = sum(r[2] == "diff" for r in _health(conn, lo, day, "reconcile"))
    return [
        ("i", "Beats both shadows (Sharpe)", r1, n1),
        ("ii", "Agent carries information (t >= 2.0, N >= 60, incr t >= 1.5)", grade(ok2), n2),
        ("iii", "No total-drawdown halt", grade(not halts), f"halts={len(halts)}"),
        ("iv", "Cost stable (<= 1.5x burn-in)", grade(ok4),
         f"per decision={'n/a' if cur is None else f'{cur:.6f}'} burn-in={'n/a' if burn is None else f'{burn:.6f}'}"),
        ("v", "Books clean (no reconcile diff in the live period)", grade(diffs == 0),
         f"diff rows in the live period={diffs} (health rows by date range, not version)"),
    ]


def _audit(conn: sqlite3.Connection, v: str, feats: pd.DataFrame, versions_path: Path, day: str | None = None) -> str:
    dates = _dates(conn, v)
    rows = _health(conn, _lo(conn, v, dates), day)
    seen = dates | {r[0][:10] for r in rows}
    done = {r[0][:10] for r in rows if r[1] == "cycle" and _detail(r[3]).get("phase") == "reconciled"}
    missed = sorted(seen - done)
    stale = sum(1 for r in rows if r[1] == "snapshot" and _detail(r[3]).get("stale"))
    gates = Counter(feats["gate_reason"].fillna("").astype(str)) if "gate_reason" in feats else Counter()
    ac = [r[0] for r in conn.execute("SELECT error FROM agent_calls WHERE substr(as_of, 1, 10) IN (%s)" % ",".join("?" * len(dates)),
                                     sorted(dates))] if dates else []
    errs = sum(1 for e in ac if e)
    return f"""- health rows are selected by date range (clock start or first decision date, to the review day); health has no version column
- missed cycles: {len(missed)} {missed[:10]} (approximation: dates with any decision or health row and no cycle row at phase "reconciled")
- stale-data halts (snapshot rows with a stale list): {stale}
- gate reason counts: {dict(gates) or 'none'}
- reconcile diffs: {sum(r[1] == 'reconcile' and r[2] == 'diff' for r in rows)}
- agent error rate: {errs}/{len(ac)}
- tags in STRATEGY_VERSIONS.md: {len(tags(versions_path))}
"""


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows yet.\n"
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _features(conn: sqlite3.Connection, version: str) -> pd.DataFrame:
    rows = [json.loads(r[0]) for r in conn.execute("SELECT row_json FROM features WHERE strategy_version = ?", (version,))]
    return pd.DataFrame(rows)


def _prices(conn: sqlite3.Connection, symbols) -> pd.DataFrame:
    syms = sorted(set(symbols))
    if not syms:
        return pd.DataFrame()
    q = f"SELECT date, symbol, adj_c FROM bars WHERE symbol IN ({','.join('?' * len(syms))})"
    df = pd.DataFrame(conn.execute(q, syms).fetchall(), columns=["date", "symbol", "adj_c"])
    return df.pivot(index="date", columns="symbol", values="adj_c").sort_index()


def default_out_dir() -> Path:
    return Path(os.environ.get("SENTRY_DATA_DIR", ROOT / "data")) / "reports"


def write_ic_daily(conn: sqlite3.Connection, strategy_version: str, day: date, seed: int = 0) -> int:
    """Upsert today's forward IC rows into ic_daily: one per (horizon, signal) with ic=mean_ic, n=n_dates,
    plus "<signal>_incr" rows (incr_ic, incr_n). Returns rows written. Same features/prices as the report."""
    feats = _features(conn, strategy_version)
    if feats.empty:
        return 0
    t = ic_table(feats, _prices(conn, feats["symbol"]), seed=seed)
    rows = []
    for r in t.itertuples(index=False):
        for suffix, ic, n in (("", r.mean_ic, r.n_dates), ("_incr", r.incr_ic, r.incr_n)):
            rows.append((day.isoformat(), r.strategy_version, int(r.horizon), r.signal + suffix,
                         None if pd.isna(ic) else float(ic), int(n)))
    conn.executemany("INSERT OR REPLACE INTO ic_daily (date, strategy_version, horizon, signal, ic, n) "
                     "VALUES (?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    return len(rows)


def render_report(conn: sqlite3.Connection, settings: Settings, strategy_version: str, day: date,
                  out_dir: Path | None = None, seed: int = 0, versions_path: Path = ROOT / "STRATEGY_VERSIONS.md") -> Path:
    out_dir = out_dir or default_out_dir()
    v = strategy_version
    trades, days, start = closed_trades_count(conn, v), trading_days(conn, v), clock_start(conn, v)
    floor = trades >= CHANGE_FLOOR_TRADES and days >= CHANGE_FLOOR_DAYS
    dec_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT substr(as_of, 1, 10) FROM decisions WHERE strategy_version = ? ORDER BY 1", (v,))]
    n_dec = conn.execute("SELECT COUNT(*) FROM decisions WHERE strategy_version = ?", (v,)).fetchone()[0]
    cost = conn.execute(f"SELECT COALESCE(SUM(usd), 0) FROM cost_daily WHERE date IN ({','.join('?' * len(dec_dates))})",
                        dec_dates).fetchone()[0] if dec_dates else 0.0
    feats = _features(conn, v)
    ic = ic_table(feats, _prices(conn, feats["symbol"]) if not feats.empty else pd.DataFrame(), seed=seed)
    shadows = pd.DataFrame(conn.execute(
        "SELECT baseline, MIN(date), MAX(date), COUNT(*) FROM shadow_equity WHERE strategy_version = ? GROUP BY baseline",
        (v,)).fetchall(), columns=["baseline", "first", "last", "days"])
    eq = conn.execute("SELECT equity, ts FROM equity_history WHERE strategy_version = ? ORDER BY ts", (v,)).fetchall()
    rets = []
    for b, first, last, _ in shadows.itertuples(index=False):
        e0, e1 = (conn.execute("SELECT equity FROM shadow_equity WHERE baseline=? AND strategy_version=? AND date=?",
                               (b, v, d)).fetchone()[0] for d in (first, last))
        rets.append(e1 / e0 - 1 if e0 else None)
    shadows["return"] = rets
    if eq:
        shadows.loc[len(shadows)] = ["real (broker equity)", eq[0][1][:10], eq[-1][1][:10],
                                     len({ts[:10] for _, ts in eq}), eq[-1][0] / eq[0][0] - 1]

    met = _metrics(_series(conn, v, start))
    crit = pd.DataFrame(evaluate_criteria(conn, v, ic, feats, met, day.isoformat()), columns=["#", "criterion", "status", "numbers"])
    burn = _cost_per_decision(conn, v, "AND substr(as_of, 1, 10) < ?", (start,)) if start else None
    cur = _cost_per_decision(conn, v, "AND substr(as_of, 1, 10) >= ?", (start,)) if start else None
    overall = ("Review threshold not met: no criterion is evaluated as evidence" if trades < REVIEW_TRADES
               else "Review threshold met: " + ("all five criteria PASS" if (crit["status"] == PASS).all()
                                                else "not all criteria PASS (see table)"))

    text = f"""# Validation report: {v}, {day.isoformat()}

**paper, forward-only, no backtest.** Every number below comes from this version's own paper rows.
It is not evidence of edge unless the review threshold is met.

**{overall}.**

## Minimum-sample status

- closed trades: {trades}; trading days since the clock start ({start or 'not started: dry run only'}): {days}
- change floor (100 closed trades AND 60 trading days): {'met' if floor else 'not met'}
- review threshold (300 closed trades): {'met' if trades >= REVIEW_TRADES else 'not met'}

## Temporal integrity

Snapshots are taken at 08:00 America/New_York with bars through the previous close, articles first seen
at least {settings.gate.news_lag_minutes} minutes before as_of, filings and macro vintages published before
as_of; every snapshot re-checks this and raises on a violation. Orders fill at 09:35, after as_of. Forward
returns in the IC table start at the close of the decision date.

## Friction model

{settings.costs.bps_per_side} bps per side plus a {settings.costs.spread_proxy_bps} bps spread proxy on traded
notional, checked against realised fills (fills.price vs fills.ref_price).

## Sample sizes and cost

- decision dates: {len(dec_dates)}; independent months: {len({d[:7] for d in dec_dates})}; decisions: {n_dec}
- LLM cost over those dates: ${cost:.4f}; per decision: ${(cost / n_dec if n_dec else 0):.6f}

- cost per decision, live (from the clock start): {_fmt(cur)}; burn-in (decision dates before the clock start): {_fmt(burn)}

## Per-series metrics (daily, from the clock start{'' if start else '; real: dry run only, nothing computed'})

{_table(met)}
Sharpe is n/a with fewer than {MIN_RETURNS} daily returns or zero std.

## Review criteria (pre-registered)

{_table(crit)}
## Forward IC (non-overlapping dates; t over dates)

{_table(ic)}
## Shadow baselines versus the real portfolio

{_table(shadows)}
## Silent-failure audit

{_audit(conn, v, feats, versions_path, day.isoformat())}"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{day.isoformat()}-{v}.md"
    path.write_text(text)
    return path


def main(argv: list[str] | None = None) -> int:
    from sentry.data.store import connect

    ap = argparse.ArgumentParser(description="Render the version-scoped paper report (on request only).")
    ap.add_argument("--version", required=True)
    ap.add_argument("--date", required=True, type=date.fromisoformat)
    ap.add_argument("--db", default=None)
    a = ap.parse_args(argv)
    print(render_report(connect(a.db), load_settings(), a.version, a.date))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
