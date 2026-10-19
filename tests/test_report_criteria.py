"""Review-criteria known answers for the version report. All data SYNTHETIC."""
import json
from datetime import date

import numpy as np
import pandas as pd

from sentry.layer_a.validation.report import (
    FAIL,
    NE,
    PASS,
    _features,
    _metrics,
    _prices,
    _series,
    evaluate_criteria,
    ic_table,
    max_drawdown,
    render_report,
    sharpe,
)
from tests.test_ic import synthetic


def load(db, shuffle):
    feats, px = synthetic(shuffle=shuffle)
    for s in px.columns:
        db.executemany("INSERT INTO bars (symbol, date, o, h, l, c, v, adj_c) VALUES (?,?,?,?,?,?,?,?)",
                       [(s, d.date().isoformat(), v, v, v, v, 1.0, v) for d, v in px[s].items()])
    db.executemany("INSERT INTO features VALUES (?,?,?,?,?)",
                   [(r.symbol, r.as_of.isoformat(), "h", r.strategy_version,
                     json.dumps({**r._asdict(), "as_of": r.as_of.isoformat()})) for r in feats.itertuples()])


V = "v1.0.0"


def test_max_drawdown_known_answer():
    assert abs(max_drawdown(pd.Series([100, 110, 99, 120])) - 0.1) < 1e-12


def test_sharpe_known_answer_and_undefined_cases():
    r = np.array([0.01, -0.01] * 15)  # 30 returns, mean 0 -> Sharpe 0
    assert abs(sharpe(pd.Series(100 * np.cumprod(np.r_[1, 1 + r])))) < 1e-9
    up = np.r_[1, 1 + np.array([0.02, 0.0] * 15)]  # mean .01, std .01*sqrt(30/29)
    assert abs(sharpe(pd.Series(np.cumprod(up))) - 0.01 / (0.01 * np.sqrt(30 / 29)) * np.sqrt(252)) < 1e-6
    assert np.isnan(sharpe(pd.Series(100 * 1.01 ** np.arange(40))))  # constant return: std 0
    assert np.isnan(sharpe(pd.Series([100, 101, 102.0])))  # < 20 returns


def build(db, real_drift, n_trades, shadow_drift=0.0, seed=1):
    """SYNTHETIC store: planted disc signal (as in test_ic), 400-bar equity curves with the given drifts."""
    load(db, shuffle=False)
    d0 = pd.bdate_range("2025-01-01", periods=400)[260:-21]
    rng = np.random.default_rng(seed)
    db.execute("INSERT INTO orders (client_order_id, broker_order_id, submitted_at, dry_run, strategy_version, strategy_hash)"
               " VALUES ('o','b',?,0,?,'h')", (d0[0].date().isoformat() + "T12:00:00+00:00", V))
    for name, drift in (("real", real_drift), ("ta_only", shadow_drift), ("random", shadow_drift)):
        eq = 100 * np.cumprod(1 + drift + rng.normal(0, 0.005, len(d0)))
        for d, e in zip(d0, eq, strict=True):
            if name == "real":
                db.execute("INSERT INTO equity_history VALUES (?,?,?)", (d.date().isoformat() + "T20:00:00+00:00", e, V))
            else:
                db.execute("INSERT INTO shadow_equity VALUES (?,?,?,?,0,0)", (d.date().isoformat(), name, V, e))
    for i in range(n_trades):
        db.execute("INSERT INTO closed_trades VALUES (?,?,?,1,1,1,1,0,'x',?,'h')", (f"T{i}", f"o{i}", f"c{i}", V))
    db.execute("UPDATE features SET row_json = json_set(row_json, '$.disc_horizon', 1)")
    db.commit()


def run(db):
    f = _features(db, V)
    return {r[0]: r for r in evaluate_criteria(db, V, ic_table(f, _prices(db, f["symbol"])), f, _metrics(_series(db, V, "2025-12-01")))}


def test_real_beats_shadows_passes_i_at_300_and_not_evaluable_at_10(db):
    build(db, real_drift=0.004, n_trades=300)
    c = run(db)
    assert c["i"][2] == PASS and c["ii"][2] == PASS and c["iii"][2] == PASS
    assert run_with_trades(db, 10)["i"][2] == NE


def run_with_trades(db, n):
    db.execute("DELETE FROM closed_trades")
    for i in range(n):
        db.execute("INSERT INTO closed_trades VALUES (?,?,?,1,1,1,1,0,'x',?,'h')", (f"T{i}", f"o{i}", f"c{i}", V))
    return run(db)


def test_real_losing_to_shadows_fails_i(db):
    build(db, real_drift=0.0, n_trades=300, shadow_drift=0.004)
    assert run(db)["i"][2] == FAIL


def test_random_disc_signal_never_passes_ii(db):
    build(db, real_drift=0.0, n_trades=300)
    load_shuffled = synthetic(shuffle=True)[0]
    db.execute("DELETE FROM features")
    db.executemany("INSERT INTO features VALUES (?,?,?,?,?)",
                   [(r.symbol, r.as_of.isoformat(), "h", V, json.dumps({**r._asdict(), "as_of": r.as_of.isoformat(), "disc_horizon": 1}))
                    for r in load_shuffled.itertuples()])
    assert run(db)["ii"][2] == FAIL  # |t| < 2 at N >= 60: evaluated and failed, never a PASS


def test_report_text_top_line_and_sections(db, settings, tmp_path):
    build(db, real_drift=0.004, n_trades=10)
    text = render_report(db, settings, V, date(2026, 10, 1), tmp_path).read_text()
    assert "Review threshold not met: no criterion is evaluated as evidence" in text
    assert "NOT EVALUABLE" in text and "Silent-failure audit" in text and "Per-series metrics" in text


def _health_row(db, ts, step, status, detail):
    db.execute("INSERT INTO health VALUES (?,?,?,?)", (ts, step, status, json.dumps(detail)))
    db.commit()


def test_reconcile_diff_in_live_period_fails_v_even_if_latest_row_is_ok(db):
    build(db, real_drift=0.004, n_trades=300)
    day = pd.bdate_range("2025-01-01", periods=400)[260:-21][5].date().isoformat()
    _health_row(db, day + "T14:00:00+00:00", "reconcile", "diff", {})
    _health_row(db, day + "T15:00:00+00:00", "reconcile", "ok", {})
    assert run(db)["v"][2] == FAIL  # SYNTHETIC: a diff stays recorded until the operator clears it


def test_total_drawdown_halt_in_range_fails_iii(db):
    build(db, real_drift=0.004, n_trades=300)
    day = pd.bdate_range("2025-01-01", periods=400)[260:-21][5].date().isoformat()
    _health_row(db, day + "T14:00:00+00:00", "risk", "halt", {"reasons": ["total drawdown 8.1% >= 8%"]})
    assert run(db)["iii"][2] == FAIL


def test_cost_per_decision_over_1_5x_burn_in_fails_iv(db):
    build(db, real_drift=0.004, n_trades=300)
    start = db.execute("SELECT MIN(substr(submitted_at, 1, 10)) FROM orders").fetchone()[0]
    for i, (d, usd) in enumerate((("2024-12-02", 1.0), (start, 10.0))):  # SYNTHETIC burn-in day then live day
        db.execute("INSERT INTO decisions VALUES (?,?,1,0.1,NULL,0.5,'{}','p',?,'h')", (f"Z{i}", f"{d}T12:00:00+00:00", V))
        db.execute("INSERT OR REPLACE INTO cost_daily VALUES (?,?)", (d, usd))
    db.commit()
    assert run(db)["iv"][2] == FAIL


def test_stale_halt_day_without_decisions_is_counted_in_audit(db, settings, tmp_path):
    build(db, real_drift=0.004, n_trades=300)
    sat = pd.bdate_range("2025-01-01", periods=400)[260:-21][3] + pd.Timedelta(days=(5 - pd.bdate_range("2025-01-01", periods=400)[260:-21][3].weekday()) % 7)
    day = sat.date().isoformat()  # SYNTHETIC Saturday: no decisions that day, inside the live period
    assert db.execute("SELECT COUNT(*) FROM decisions WHERE substr(as_of, 1, 10) = ?", (day,)).fetchone()[0] == 0
    _health_row(db, day + "T13:00:00+00:00", "snapshot", "halt", {"stale": ["AAA"]})
    text = render_report(db, settings, V, date(2026, 10, 1), tmp_path).read_text()
    assert "stale-data halts (snapshot rows with a stale list): 1" in text
