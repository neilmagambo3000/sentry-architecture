"""Live ingest callable for run_cycle step 2. Idempotent: store PKs make
re-inserts safe. A loader failure is caught and reported per source; the stale-data guard decides."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")
BARS_BACK_DAYS = 760


def make_ingest(bars_loader, news_loader, edgar_loader, alfred_loader, symbols, macro_series, clock,
                news_window_days: int = 7):
    symbols, edgar_done = list(symbols), set()

    def _val(conn, sql, *args):
        return conn.execute(sql, args).fetchone()[0]

    def ingest(conn, as_of: datetime) -> dict:
        day = as_of.astimezone(NY).date()
        report: dict = {}

        def step(name, fn):
            try:
                report[name] = {"status": "ok", "count": fn()}
            except Exception as e:  # noqa: BLE001 - one vendor outage must not crash the loop
                report[name] = {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}

        def bars():
            end, n = day - timedelta(days=1), 0  # previous trading day or earlier; vendors return only real bars
            for s in symbols:
                last = _val(conn, "SELECT MAX(date) FROM bars WHERE symbol = ?", s)
                start = (datetime.fromisoformat(last).date() + timedelta(days=1)) if last else day - timedelta(days=BARS_BACK_DAYS)
                if start <= end:
                    n += bars_loader.load(conn, [s], start, end)
            return n

        def news():
            last = _val(conn, "SELECT MAX(first_seen) FROM articles")
            start = datetime.fromisoformat(last) if last else as_of - timedelta(days=news_window_days)
            arrival = clock()
            return news_loader.load(conn, symbols, start, arrival, arrival_time=arrival)

        def edgar():
            if day in edgar_done:
                return 0
            n = edgar_loader.load(conn, symbols)
            edgar_done.add(day)
            return n

        def alfred():
            n = 0
            for sid in macro_series.values():
                last = _val(conn, "SELECT MAX(vintage) FROM macro_vintages WHERE series = ?", sid)
                since = datetime.fromisoformat(last).date() if last else day - timedelta(days=BARS_BACK_DAYS)
                n += alfred_loader.load(conn, sid, since)
            return n

        for name, fn in (("bars", bars), ("news", news), ("edgar", edgar), ("alfred", alfred)):
            step(name, fn)
        for k, v in report.items():
            if v["status"] == "error":  # no health row: "ingest" is not a health step (the e2e test pins STEPS)
                log.warning("ingest %s failed: %s", k, v["error"])
        return report

    return ingest
