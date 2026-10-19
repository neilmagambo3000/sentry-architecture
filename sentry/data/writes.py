"""Write helpers into the store. Used by the loaders and by tests.

Timestamps are written as ISO-8601 UTC strings so lexical order equals time order.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone

import pandas as pd

from sentry.types import Article, Filing, MacroVintage

BAR_COLUMNS = ["open", "high", "low", "close", "volume", "adj_close"]


def iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError(f"naive datetime {dt!r}: store timestamps must be tz-aware")
    return dt.astimezone(timezone.utc).isoformat()


def _day(d) -> str:
    return pd.Timestamp(d).date().isoformat()


def insert_bars(conn: sqlite3.Connection, symbol: str, df: pd.DataFrame) -> int:
    """df: index date-like, columns open high low close volume adj_close. Upserts by (symbol, date)."""
    missing = set(BAR_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"insert_bars: missing columns {sorted(missing)}")
    rows = [
        (symbol, _day(d), float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume), float(r.adj_close))
        for d, r in zip(df.index, df[BAR_COLUMNS].itertuples(index=False), strict=True)
    ]
    conn.executemany("INSERT OR REPLACE INTO bars (symbol, date, o, h, l, c, v, adj_c) VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    return len(rows)


def insert_article(conn: sqlite3.Connection, a: Article, *, commit: bool = True) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO articles (id, first_seen, published, source, headline, body, url) VALUES (?,?,?,?,?,?,?)",
        (a.id, iso_utc(a.first_seen), iso_utc(a.published) if a.published else None, a.source, a.headline, a.body, a.url),
    )
    conn.executemany(
        "INSERT OR IGNORE INTO article_symbols (article_id, symbol) VALUES (?,?)", [(a.id, s) for s in a.symbols]
    )
    if commit:
        conn.commit()


def insert_filing(conn: sqlite3.Connection, f: Filing, *, commit: bool = True) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO filings (id, symbol, form, filed_at, period_end, facts_json) VALUES (?,?,?,?,?,?)",
        (f.id, f.symbol, f.form, iso_utc(f.filed_at), f.period_end.isoformat(), json.dumps(dict(f.facts), sort_keys=True)),
    )
    if commit:
        conn.commit()


def insert_macro(conn: sqlite3.Connection, m: MacroVintage, *, commit: bool = True) -> None:
    obs = {(d.isoformat() if isinstance(d, date) else str(d)): float(v) for d, v in m.observations.items()}
    conn.execute(
        "INSERT OR REPLACE INTO macro_vintages (series, vintage, obs_json) VALUES (?,?,?)",
        (m.series, iso_utc(m.vintage), json.dumps(obs, sort_keys=True)),
    )
    if commit:
        conn.commit()
