"""Point-in-time contract of SqliteSnapshotSource. All data here is synthetic."""
import time
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from sentry.data.snapshot import PointInTimeError, SqliteSnapshotSource, check_point_in_time
from sentry.data.store import connect
from sentry.data.writes import insert_article, insert_bars, insert_filing, insert_macro
from sentry.types import Article, Filing, MacroVintage

UTC, NY = timezone.utc, ZoneInfo("America/New_York")
AS_OF = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)  # 08:00 ET


def bars_df(dates, base=100.0):
    n = len(dates)
    c = base + np.arange(n, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1e6, "adj_close": c},
                        index=pd.DatetimeIndex(dates))


def art(i, first_seen, symbols=("AAA",)):
    return Article(f"a{i}", tuple(symbols), first_seen, first_seen, "synthetic", f"headline {i}", None, "")


def filing(i, filed_at, sym="AAA"):
    return Filing(f"f{i}", sym, "10-Q", filed_at, date(2026, 6, 30), {"revenue": 1.0, "months": 6.0})


@pytest.fixture
def src(db, settings):
    return SqliteSnapshotSource(db, settings.gate)


def test_naive_as_of_raises(src):
    with pytest.raises(ValueError):
        src.get("AAA", datetime(2026, 9, 30, 12))


def test_bar_for_today_invisible_before_close(db, src):
    insert_bars(db, "AAA", bars_df(["2026-09-28", "2026-09-29", "2026-09-30"]))
    before = datetime(2026, 9, 30, 15, 59, tzinfo=NY)
    at_close = datetime(2026, 9, 30, 16, 0, tzinfo=NY)
    assert src.get("AAA", before).bars.index[-1] == pd.Timestamp("2026-09-29")
    assert src.get("AAA", at_close).bars.index[-1] == pd.Timestamp("2026-09-30")
    assert list(src.get("AAA", AS_OF).bars.columns) == ["open", "high", "low", "close", "volume", "adj_close"]


def test_article_lag_and_window(db, src):
    insert_article(db, art(1, AS_OF - timedelta(minutes=15)))  # inside the 30-minute lag: invisible
    insert_article(db, art(2, AS_OF - timedelta(minutes=30)))  # exactly at the lag: visible
    insert_article(db, art(3, AS_OF - timedelta(days=6)))  # outside the 5-day window: invisible
    insert_article(db, art(4, AS_OF - timedelta(hours=2), symbols=("AAA", "BBB")))
    insert_article(db, art(5, AS_OF + timedelta(days=1)))  # future: invisible
    snap = src.get("AAA", AS_OF)
    assert [a.id for a in snap.articles] == ["a4", "a2"]
    assert snap.articles[0].symbols == ("AAA", "BBB")
    assert [a.id for a in src.get("BBB", AS_OF).articles] == ["a4"]


def test_filing_one_minute_after_as_of_is_invisible(db, src):
    insert_filing(db, filing(1, AS_OF + timedelta(minutes=1)))
    insert_filing(db, filing(2, AS_OF))
    snap = src.get("AAA", AS_OF)
    assert [f.id for f in snap.filings] == ["f2"]
    assert snap.filings[0].facts == {"revenue": 1.0, "months": 6.0}


def test_macro_latest_visible_vintage(db, src):
    for d, v in [(1, 4.0), (2, 4.1), (3, 4.2)]:
        insert_macro(db, MacroVintage("UNRATE", AS_OF + timedelta(days=d - 2), {date(2026, 8, 1): v}))
    m = src.get("AAA", AS_OF).macro["UNRATE"]
    assert m.vintage == AS_OF and m.observations == {date(2026, 8, 1): 4.1}


def _clean_snapshot(src, db):
    insert_bars(db, "AAA", bars_df(["2026-09-28", "2026-09-29"]))
    insert_article(db, art(1, AS_OF - timedelta(hours=1)))
    insert_filing(db, filing(1, AS_OF - timedelta(days=1)))
    insert_macro(db, MacroVintage("UNRATE", AS_OF - timedelta(days=1), {date(2026, 8, 1): 4.0}))
    return src.get("AAA", AS_OF)


def test_planted_violations_raise(db, src, settings):
    snap = _clean_snapshot(src, db)
    lag = settings.gate.news_lag_minutes
    check_point_in_time(snap, lag)  # clean snapshot passes
    planted = [
        replace(snap, articles=snap.articles + (art(9, AS_OF - timedelta(minutes=1)),)),
        replace(snap, filings=snap.filings + (filing(9, AS_OF + timedelta(minutes=1)),)),
        replace(snap, macro={"UNRATE": MacroVintage("UNRATE", AS_OF + timedelta(seconds=1), {})}),
        replace(snap, bars=bars_df(["2026-09-30"])),  # today's bar at 08:00 ET
    ]
    for bad in planted:
        with pytest.raises(PointInTimeError):
            check_point_in_time(bad, lag)


def test_bypassed_sql_filter_raises(db, src, monkeypatch):
    insert_article(db, art(1, AS_OF + timedelta(hours=3)))  # future row in the store
    unfiltered = lambda symbol, as_of: (art(1, AS_OF + timedelta(hours=3)),)  # noqa: E731
    monkeypatch.setattr(src, "_articles", unfiltered)
    with pytest.raises(PointInTimeError):
        src.get("AAA", AS_OF)


def test_universe_day_replays_fast(tmp_path, settings):
    """Performance check: 20 synthetic symbols x ~2 years of bars from on-disk SQLite in < 2 s."""
    conn = connect(tmp_path / "sentry.db")
    days = pd.bdate_range(end="2026-09-29", periods=504)
    syms = [f"S{i:02d}" for i in range(20)]
    for i, s in enumerate(syms):
        insert_bars(conn, s, bars_df(days, base=50 + i))
        for k in range(10):
            insert_article(conn, art(f"{s}-{k}", AS_OF - timedelta(hours=5 + k), symbols=(s,)), commit=False)
        for q in range(6):
            insert_filing(conn, filing(f"{s}-{q}", AS_OF - timedelta(days=30 + 91 * q), sym=s), commit=False)
    insert_macro(conn, MacroVintage("UNRATE", AS_OF - timedelta(days=3), {date(2026, 8, 1): 4.0}))
    src = SqliteSnapshotSource(conn, settings.gate)
    t0 = time.perf_counter()
    out = src.universe_day(syms, AS_OF)
    elapsed = time.perf_counter() - t0
    assert len(out) == 20 and all(len(s.bars) == 504 and len(s.filings) == 6 for s in out.values())
    assert elapsed < 2.0, elapsed
    conn.close()
