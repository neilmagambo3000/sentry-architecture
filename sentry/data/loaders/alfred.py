"""ALFRED vintages -> store (write-only). Network access only through the injected fredapi.Fred.

`get_series_all_releases` returns every (observation date, realtime_start, value). One MacroVintage is
written per distinct realtime_start >= `since`: for each observation date, the latest value released on
or before that day. FRED gives only the release date, so the vintage time is set conservatively to
`release_hour_et`:00 New York time on that day (after the 08:30 ET releases). To keep rows small each
vintage holds only observations dated within `lookback_days` of the release.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from sentry.data.writes import insert_macro
from sentry.types import MacroVintage

NY = ZoneInfo("America/New_York")


class AlfredLoader:
    def __init__(self, fred):
        self.fred = fred  # fredapi.Fred or a fake with get_series_all_releases

    @classmethod
    def from_env(cls) -> "AlfredLoader":
        from fredapi import Fred

        return cls(Fred(api_key=os.environ["FRED_API_KEY"]))

    def load(self, conn: sqlite3.Connection, series_id: str, since: date, lookback_days: int = 400,
             release_hour_et: int = 17) -> int:
        # Bounded real-time window: unbounded requests exceed FRED's 2000-vintage-date limit. Observations dated
        # within `lookback_days` of a vintage are first released on or after (since - lookback_days).
        df = self.fred.get_series_all_releases(
            series_id, realtime_start=(since - timedelta(days=lookback_days)).isoformat(),
            realtime_end=date.today().isoformat())
        df = df.assign(date=pd.to_datetime(df["date"]).dt.date,
                       realtime_start=pd.to_datetime(df["realtime_start"]).dt.date).dropna(subset=["value"])
        df = df.sort_values(["realtime_start", "date"])
        n = 0
        for rt in sorted(d for d in df["realtime_start"].unique() if d >= since):
            known = df[(df["realtime_start"] <= rt) & (df["date"] >= rt - timedelta(days=lookback_days))]
            latest = known.groupby("date")["value"].last()  # sorted by realtime_start, so last = latest release
            insert_macro(conn, MacroVintage(
                series=series_id,
                vintage=datetime.combine(rt, time(release_hour_et), tzinfo=NY),
                observations={d: float(v) for d, v in latest.items()},
            ), commit=False)
            n += 1
        conn.commit()
        return n
