"""Alpaca daily bars -> store (write-only). Network access only through the injected client.

Two requests per load: raw OHLCV and all-adjusted closes (splits + dividends). Note that vendor
adjustment is as of fetch time; TA uses returns of adj_close, which later dividends/splits rescale
uniformly, so returns are unaffected.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from sentry.data.writes import insert_bars

NY = ZoneInfo("America/New_York")


class AlpacaBarsLoader:
    def __init__(self, client):
        self.client = client  # alpaca.data.historical.StockHistoricalDataClient or a fake with get_stock_bars

    @classmethod
    def from_env(cls) -> "AlpacaBarsLoader":
        from alpaca.data.historical import StockHistoricalDataClient

        return cls(StockHistoricalDataClient(os.environ["ALPACA_PAPER_KEY_ID"], os.environ["ALPACA_PAPER_SECRET"]))

    def _fetch(self, symbols: list[str], start: date, end: date, adjustment: str) -> dict[str, list]:
        from alpaca.data.enums import Adjustment
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        req = StockBarsRequest(
            symbol_or_symbols=symbols, timeframe=TimeFrame.Day, adjustment=Adjustment(adjustment),
            start=datetime.combine(start, time(0), tzinfo=NY), end=datetime.combine(end, time(23, 59), tzinfo=NY),
        )
        return self.client.get_stock_bars(req).data

    def load(self, conn: sqlite3.Connection, symbols: list[str], start: date, end: date) -> int:
        raw = self._fetch(symbols, start, end, "raw")
        adj = self._fetch(symbols, start, end, "all")
        n = 0
        for sym in symbols:
            adj_close = {b.timestamp.astimezone(NY).date(): b.close for b in adj.get(sym, [])}
            rows = [
                (b.timestamp.astimezone(NY).date(), b.open, b.high, b.low, b.close, b.volume,
                 adj_close.get(b.timestamp.astimezone(NY).date()))
                for b in raw.get(sym, [])
            ]
            rows = [r for r in rows if r[-1] is not None]  # a bar without an adjusted close is not stored
            if rows:
                df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "adj_close"])
                n += insert_bars(conn, sym, df.set_index("date"))
        return n
