"""Forward IC known answers: a planted signal shows up, a shuffled one does not.
Also renders the version-scoped report into a temp dir. All data SYNTHETIC."""
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from sentry.layer_a.validation.baselines import momentum_12_1, random_signal
from sentry.layer_a.validation.ic import ic_table

N_SYM, N_DAYS = 30, 400


def synthetic(seed=3, shuffle=False):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=N_DAYS)
    syms = [f"S{i:02d}" for i in range(N_SYM)]
    px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (N_DAYS, N_SYM)), axis=0), index=dates, columns=syms)
    fwd1 = (px.shift(-1) / px - 1)
    rows = []
    for d in dates[260:-21]:  # decision dates with 12-1 momentum history and forward returns
        z = fwd1.loc[d].to_numpy() * 100 + rng.normal(0, 1.0, N_SYM)  # planted: knows tomorrow's return, noisily
        if shuffle:
            z = rng.permutation(z)  # within-date shuffle destroys the link and keeps the distribution
        for s, v in zip(syms, z, strict=True):
            rows.append({"symbol": s, "as_of": datetime.combine(d.date(), datetime.min.time(), timezone.utc)
                         + timedelta(hours=12), "strategy_version": "v1.0.0", "disc_direction": int(np.sign(v)),
                         "disc_confidence": min(1.0, abs(v) / 3), "control_1": rng.normal(), "control_2": rng.normal()})
    return pd.DataFrame(rows), px


def get(t, sig, h):
    return t[(t.signal == sig) & (t.horizon == h)].iloc[0]


def test_planted_signal_has_positive_ic_with_large_t():
    feats, px = synthetic()
    t = ic_table(feats, px, horizons=(1, 5))
    r = get(t, "disc", 1)
    assert r.mean_ic > 0.2 and r.t_stat > 10 and r.n_dates == len(feats.as_of.unique())
    assert r.incr_ic > 0.2 and r.incr_t > 10  # controls are noise, so the incremental IC survives
    assert get(t, "disc", 5).n_dates == len(range(0, r.n_dates, 5))  # non-overlapping dates only


def test_shuffled_signal_has_no_ic():
    feats, px = synthetic(shuffle=True)
    t = ic_table(feats, px, horizons=(1,))
    r = get(t, "disc", 1)
    assert abs(r.t_stat) < 2 and abs(r.incr_t) < 2


def test_baselines_in_table_and_no_edge_on_random_walk():
    feats, px = synthetic()
    t = ic_table(feats, px, horizons=(1,))
    assert set(t.signal) == {"disc", "random", "mom_12_1"} and set(t.strategy_version) == {"v1.0.0"}
    for sig in ("random", "mom_12_1"):
        assert abs(get(t, sig, 1).t_stat) < 2  # random-walk prices: neither baseline should find anything


def test_baselines_known_answers():
    px = pd.DataFrame({"A": np.arange(1.0, 301.0)}, index=pd.bdate_range("2025-01-01", periods=300))
    f = pd.DataFrame({"symbol": ["A", "A"], "date": [px.index[260], px.index[100]]})
    m = momentum_12_1(f, px)
    t = 259  # last close strictly before the decision date
    assert m.iloc[0] == pytest.approx(px.A.iloc[t - 21] / px.A.iloc[t - 252] - 1) and np.isnan(m.iloc[1])
    r1, r2 = random_signal(f, 1), random_signal(f.iloc[::-1], 1)
    assert r1.to_dict() == r2.to_dict() and r1.between(-0.5, 0.5).all()


def test_constant_signal_has_no_ic_not_float_noise():
    feats, px = synthetic()
    feats["disc_direction"], feats["disc_confidence"] = 1, 0.7  # a canned view: identical for every symbol
    r = get(ic_table(feats, px, horizons=(1,)), "disc", 1)
    assert r.n_dates == 0 and r.incr_n == 0


def test_grouped_by_version_and_empty():
    feats, px = synthetic()
    feats.loc[feats.index[: len(feats) // 2], "strategy_version"] = "v0.9.0"
    assert set(ic_table(feats, px, horizons=(1,)).strategy_version) == {"v0.9.0", "v1.0.0"}
    assert ic_table(feats.iloc[:0], px).empty


def test_report_renders_labelled_paper_forward_only(tmp_path, db, settings):
    from sentry.layer_a.validation.report import render_report

    feats, px = synthetic()
    for s in px.columns:
        db.executemany("INSERT INTO bars (symbol, date, o, h, l, c, v, adj_c) VALUES (?,?,?,?,?,?,?,?)",
                       [(s, d.date().isoformat(), v, v, v, v, 1.0, v) for d, v in px[s].items()])
    import json

    db.executemany("INSERT INTO features VALUES (?,?,?,?,?)",
                   [(r.symbol, r.as_of.isoformat(), "h", r.strategy_version,
                     json.dumps({**r._asdict(), "as_of": r.as_of.isoformat()})) for r in feats.itertuples()])
    path = render_report(db, settings, "v1.0.0", date(2026, 10, 1), out_dir=tmp_path)
    text = path.read_text()
    assert path.name == "2026-10-01-v1.0.0.md"
    head = text.split("## Temporal integrity")[0]
    assert "paper, forward-only, no backtest" in head and "Minimum-sample status" in head
    assert "change floor (100 closed trades AND 60 trading days): not met" in head
    assert "| disc | 1 |" in text and chr(0x2014) not in text  # no em dashes in generated docs
