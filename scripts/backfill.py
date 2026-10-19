"""Human-run: load ~2 years of bars, recent news, EDGAR facts and ALFRED vintages into the store, once, so
the rest of the pipeline has data. Free data only, but it needs the vendor
keys from the environment (the same loaders the scheduler builds with from_env()). Not run by tests or CI.

Usage: .venv/bin/python scripts/backfill.py --symbols AAA,BBB [--as-of 2026-10-01T08:00] [--db data/sentry.db]
   or: .venv/bin/python scripts/backfill.py --symbols AAPL,MSFT,SPY
Idempotent (store PKs); re-running only fetches what is newer than the store.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentry.data.store import connect  # noqa: E402
from sentry.live.ingest import make_ingest  # noqa: E402

MACRO_SERIES = {"unemployment": "UNRATE"}  # example FRED series id
UTC, NY = timezone.utc, ZoneInfo("America/New_York")


def backfill(conn, loaders, symbols, macro_series, as_of: datetime, news_days: int = 30, clock=None) -> dict:
    """loaders = (bars, news, edgar, alfred). Returns make_ingest's per-source report."""
    clock = clock or (lambda: datetime.now(UTC))
    return make_ingest(*loaders, symbols, macro_series, clock, news_days)(conn, as_of)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sectors", type=Path, help="csv with header symbol,sector")
    src.add_argument("--symbols", help="comma separated list")
    ap.add_argument("--as-of", type=datetime.fromisoformat, default=None, help="ISO; naive means America/New_York")
    ap.add_argument("--news-days", type=int, default=30)
    ap.add_argument("--db", default=None)
    a = ap.parse_args(argv)
    if a.sectors:
        with a.sectors.open() as fh:
            symbols = [r["symbol"].strip() for r in csv.DictReader(fh)]
    else:
        symbols = [s.strip() for s in a.symbols.split(",") if s.strip()]
    as_of = a.as_of or datetime.now(UTC)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=NY)

    from sentry.data.loaders.alfred import AlfredLoader
    from sentry.data.loaders.alpaca_bars import AlpacaBarsLoader
    from sentry.data.loaders.alpaca_news import AlpacaNewsLoader
    from sentry.data.loaders.edgar import EdgarLoader

    loaders = (AlpacaBarsLoader.from_env(), AlpacaNewsLoader.from_env(), EdgarLoader.from_env(),
               AlfredLoader.from_env())
    report = backfill(connect(a.db), loaders, symbols, MACRO_SERIES, as_of, a.news_days)
    for name, r in report.items():
        print(f"{name}: {r['status']} " + (f"({r['count']} rows)" if r["status"] == "ok" else r["error"]))
    return 0 if all(r["status"] == "ok" for r in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
