"""Snapshot accessor: the only read path for market, news, filing and macro data.

Visibility rules for a tz-aware `as_of`:
- bars: a bar dated D is visible iff as_of >= D 16:00 America/New_York (the close has happened)
- articles: first_seen + news_lag_minutes <= as_of and first_seen >= as_of - news_window_days
- filings: filed_at <= as_of
- macro: per series, the latest vintage with vintage <= as_of

SQL does the filtering; `check_point_in_time` then re-verifies every item in Python and raises
`PointInTimeError` (an explicit raise, so `python -O` cannot strip it).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from sentry.config import GateConfig
from sentry.data.writes import BAR_COLUMNS, iso_utc
from sentry.types import Article, Filing, MacroVintage, Snapshot

NY = ZoneInfo("America/New_York")
CLOSE = time(16, 0)


class PointInTimeError(Exception):
    """A snapshot contains something that was not knowable at its as_of."""


def _require_aware(as_of: datetime) -> None:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError(f"as_of must be tz-aware, got {as_of!r}")


def bar_close_time(d: date) -> datetime:
    """The moment a daily bar dated `d` becomes known (16:00 New York time)."""
    return datetime.combine(d, CLOSE, tzinfo=NY)


def last_visible_bar_date(as_of: datetime) -> date:
    local = as_of.astimezone(NY)
    return local.date() if local.time() >= CLOSE else local.date() - timedelta(days=1)


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def check_point_in_time(snap: Snapshot, news_lag_minutes: int) -> None:
    """Raise PointInTimeError if any bar, article, filing or macro vintage in `snap` is from after its as_of."""
    as_of, lag = snap.as_of, timedelta(minutes=news_lag_minutes)
    for d in snap.bars.index:
        if bar_close_time(pd.Timestamp(d).date()) > as_of:
            raise PointInTimeError(f"{snap.symbol}: bar {d} closes after as_of {as_of}")
    for a in snap.articles:
        if a.first_seen + lag > as_of:
            raise PointInTimeError(f"{snap.symbol}: article {a.id} first_seen {a.first_seen} + lag > as_of {as_of}")
    for f in snap.filings:
        if f.filed_at > as_of:
            raise PointInTimeError(f"{snap.symbol}: filing {f.id} filed_at {f.filed_at} > as_of {as_of}")
    for s, m in snap.macro.items():
        if m.vintage > as_of:
            raise PointInTimeError(f"macro {s}: vintage {m.vintage} > as_of {as_of}")


class SqliteSnapshotSource:
    def __init__(self, conn: sqlite3.Connection, gate_cfg: GateConfig):
        self.conn = conn
        self.cfg = gate_cfg

    # ------------------------------------------------------------ queries (filtering lives here)

    def _bars(self, symbol: str, as_of: datetime) -> pd.DataFrame:
        rows = self.conn.execute(
            "SELECT date, o, h, l, c, v, adj_c FROM bars WHERE symbol = ? AND date <= ? ORDER BY date",
            (symbol, last_visible_bar_date(as_of).isoformat()),
        ).fetchall()
        df = pd.DataFrame([tuple(r)[1:] for r in rows], columns=BAR_COLUMNS, dtype=float)
        df.index = pd.DatetimeIndex([r[0] for r in rows], name="date")
        return df

    def _articles(self, symbol: str, as_of: datetime) -> tuple[Article, ...]:
        hi = iso_utc(as_of - timedelta(minutes=self.cfg.news_lag_minutes))
        lo = iso_utc(as_of - timedelta(days=self.cfg.news_window_days))
        rows = self.conn.execute(
            "SELECT a.id, a.first_seen, a.published, a.source, a.headline, a.body, a.url "
            "FROM articles a JOIN article_symbols s ON s.article_id = a.id "
            "WHERE s.symbol = ? AND a.first_seen <= ? AND a.first_seen >= ? ORDER BY a.first_seen, a.id",
            (symbol, hi, lo),
        ).fetchall()
        if not rows:
            return ()
        ids = [r["id"] for r in rows]
        syms: dict[str, list[str]] = {i: [] for i in ids}
        q = f"SELECT article_id, symbol FROM article_symbols WHERE article_id IN ({','.join('?' * len(ids))}) ORDER BY symbol"
        for aid, s in self.conn.execute(q, ids).fetchall():
            syms[aid].append(s)
        return tuple(
            Article(r["id"], tuple(syms[r["id"]]), _dt(r["first_seen"]), _dt(r["published"]), r["source"],
                    r["headline"], r["body"], r["url"])
            for r in rows
        )

    def _filings(self, symbol: str, as_of: datetime) -> tuple[Filing, ...]:
        rows = self.conn.execute(
            "SELECT id, symbol, form, filed_at, period_end, facts_json FROM filings "
            "WHERE symbol = ? AND filed_at <= ? ORDER BY filed_at, id",
            (symbol, iso_utc(as_of)),
        ).fetchall()
        return tuple(
            Filing(r["id"], r["symbol"], r["form"], _dt(r["filed_at"]), date.fromisoformat(r["period_end"]),
                   json.loads(r["facts_json"]))
            for r in rows
        )

    def _macro(self, as_of: datetime) -> dict[str, MacroVintage]:
        t = iso_utc(as_of)
        rows = self.conn.execute(
            "SELECT m.series, m.vintage, m.obs_json FROM macro_vintages m "
            "WHERE m.vintage = (SELECT MAX(vintage) FROM macro_vintages WHERE series = m.series AND vintage <= ?)",
            (t,),
        ).fetchall()
        return {
            r["series"]: MacroVintage(
                r["series"], _dt(r["vintage"]),
                {date.fromisoformat(k): v for k, v in sorted(json.loads(r["obs_json"]).items())},
            )
            for r in rows
        }

    # ------------------------------------------------------------ public API

    def _build(self, symbol: str, as_of: datetime, macro: dict[str, MacroVintage]) -> Snapshot:
        snap = Snapshot(symbol, as_of.astimezone(timezone.utc), self._bars(symbol, as_of),
                        self._articles(symbol, as_of), self._filings(symbol, as_of), macro)
        check_point_in_time(snap, self.cfg.news_lag_minutes)
        return snap

    def get(self, symbol: str, as_of: datetime) -> Snapshot:
        _require_aware(as_of)
        return self._build(symbol, as_of, self._macro(as_of))

    def universe_day(self, symbols: list[str], as_of: datetime) -> dict[str, Snapshot]:
        _require_aware(as_of)
        macro = self._macro(as_of)  # shared across symbols
        return {s: self._build(s, as_of, macro) for s in symbols}
