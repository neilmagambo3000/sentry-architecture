"""Loaders with fake vendor clients only (no network, from_env never called). All data is synthetic."""
from datetime import date, datetime, timezone
from types import SimpleNamespace as NS
from zoneinfo import ZoneInfo

import pandas as pd

from sentry.data.loaders.alfred import AlfredLoader
from sentry.data.loaders.alpaca_bars import AlpacaBarsLoader
from sentry.data.loaders.alpaca_news import AlpacaNewsLoader
from sentry.data.loaders.edgar import EdgarLoader, filings_from_companyfacts
from sentry.data.snapshot import SqliteSnapshotSource

UTC, NY = timezone.utc, ZoneInfo("America/New_York")


class FakeBars:
    def get_stock_bars(self, req):
        k = 0.5 if req.adjustment.value == "all" else 1.0  # adjusted request returns halved prices
        ts = [datetime(2026, 9, d, 4, 0, tzinfo=UTC) for d in (28, 29)]  # Alpaca daily bars stamp 00:00 ET
        return NS(data={"AAA": [NS(timestamp=t, open=10 * k, high=11 * k, low=9 * k, close=(10 + i) * k, volume=100)
                                for i, t in enumerate(ts)]})


def test_alpaca_bars_loader(db):
    n = AlpacaBarsLoader(FakeBars()).load(db, ["AAA"], date(2026, 9, 28), date(2026, 9, 29))
    assert n == 2
    rows = db.execute("SELECT date, c, adj_c FROM bars WHERE symbol='AAA' ORDER BY date").fetchall()
    assert [tuple(r) for r in rows] == [("2026-09-28", 10.0, 5.0), ("2026-09-29", 11.0, 5.5)]


def news_item(i, created):
    return NS(id=i, symbols=["AAA", "BBB"], created_at=created, source="synthetic", headline=f"h{i}",
              content=f"body {i}", summary="", url=f"https://example.invalid/{i}")


class FakeNews:
    def __init__(self):
        self.pages = {None: ([news_item(1, datetime(2026, 9, 29, 13, tzinfo=UTC))], "tok"),
                      "tok": ([news_item(2, datetime(2026, 9, 29, 14, tzinfo=UTC))], None)}

    def get_news(self, req):
        items, nxt = self.pages[req.page_token]
        return NS(data={"news": items}, next_page_token=nxt)


def test_alpaca_news_historical_and_live_first_seen(db):
    start, end = datetime(2026, 9, 29, tzinfo=UTC), datetime(2026, 9, 30, tzinfo=UTC)
    assert AlpacaNewsLoader(FakeNews()).load(db, ["AAA"], start, end) == 2
    rows = db.execute("SELECT id, first_seen, published FROM articles ORDER BY id").fetchall()
    assert rows[0]["first_seen"] == rows[0]["published"] == "2026-09-29T13:00:00+00:00"
    arrival = datetime(2026, 9, 29, 18, tzinfo=UTC)
    AlpacaNewsLoader(FakeNews()).load(db, ["AAA"], start, end, arrival_time=arrival)
    assert db.execute("SELECT first_seen FROM articles WHERE id='1'").fetchone()[0] == "2026-09-29T18:00:00+00:00"
    assert {r[0] for r in db.execute("SELECT symbol FROM article_symbols WHERE article_id='2'")} == {"AAA", "BBB"}


def e(val, end, accn, form="10-Q", filed="2026-08-01", start=None):
    d = {"val": val, "end": end, "accn": accn, "form": form, "filed": filed}
    if start:
        d["start"] = start
    return d


COMPANYFACTS = {  # synthetic SEC companyfacts payload
    "facts": {
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [e(100, "2026-07-25", "A2")]}}},
        "us-gaap": {
            "Revenues": {"units": {"USD": [
                e(250, "2026-06-30", "A2", start="2026-04-01"),  # quarter
                e(500, "2026-06-30", "A2", start="2026-01-01"),  # YTD: kept
                e(240, "2026-03-31", "A1", filed="2026-05-01", start="2026-01-01"),
                e(999, "2026-03-31", "A2", start="2026-01-01"),  # comparative in A2: not first-reported for Q1
                e(777, "2026-06-30", "A3", form="10-Q/A", filed="2026-09-01", start="2026-01-01"),  # amendment
            ]}},
            "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
                e(60, "2026-06-30", "A2", start="2026-01-01")]}},
            "OperatingIncomeLoss": {"units": {"USD": [
                e(35, "2026-06-30", "A2", start="2026-04-01"), e(70, "2026-06-30", "A2", start="2026-01-01")]}},
            "DepreciationDepletionAndAmortization": {"units": {"USD": [
                e(10, "2026-06-30", "A2", start="2026-01-01")]}},
            "EarningsPerShareDiluted": {"units": {"USD/shares": [e(0.5, "2026-06-30", "A2", start="2026-04-01")]}},
            "Assets": {"units": {"USD": [e(1000, "2026-06-30", "A2"), e(900, "2025-12-31", "A2")]}},
        },
    }
}


def test_edgar_first_reported_ytd_facts():
    fs = {f.id: f for f in filings_from_companyfacts("AAA", COMPANYFACTS)}
    assert set(fs) == {"A1", "A2"}  # amendment skipped
    a2 = fs["A2"]
    assert a2.period_end == date(2026, 6, 30) and a2.form == "10-Q"
    assert a2.filed_at == datetime(2026, 8, 1, 17, 30, tzinfo=NY)
    assert a2.facts == {"months": 6.0, "revenue": 500.0, "operating_cash_flow": 60.0, "operating_income": 70.0,
                        "d_and_a": 10.0, "ebitda": 80.0, "total_assets": 1000.0, "shares_outstanding": 100.0}
    # quarterly-only EPS (3 months) differs from the filing's YTD length: dropped, not mixed in
    assert fs["A1"].facts == {"months": 3.0, "revenue": 240.0}


def test_edgar_loader_writes(db):
    client = NS(company_facts=lambda ticker: COMPANYFACTS)
    assert EdgarLoader(client).load(db, ["AAA"]) == 2
    as_of = datetime(2026, 8, 2, tzinfo=UTC)
    from sentry.config import GateConfig
    src = SqliteSnapshotSource(db, GateConfig(30, 5))
    assert [f.id for f in src.get("AAA", as_of).filings] == ["A1", "A2"]
    assert [f.id for f in src.get("AAA", datetime(2026, 8, 1, 21, 29, tzinfo=UTC)).filings] == ["A1"]


class FakeFred:
    def get_series_all_releases(self, series_id, realtime_start=None, realtime_end=None):
        return pd.DataFrame({
            "date": pd.to_datetime(["2026-06-01", "2026-07-01", "2026-06-01", "2026-08-01"]),
            "realtime_start": pd.to_datetime(["2026-07-03", "2026-08-07", "2026-08-07", "2026-09-05"]),
            "value": [4.1, 4.2, 4.0, 4.3],
        })


def test_alfred_vintages(db, settings):
    assert AlfredLoader(FakeFred()).load(db, "UNRATE", since=date(2026, 1, 1)) == 3
    src = SqliteSnapshotSource(db, settings.gate)
    m = src.get("AAA", datetime(2026, 8, 10, tzinfo=UTC)).macro["UNRATE"]
    assert m.vintage == datetime(2026, 8, 7, 17, tzinfo=NY)
    assert m.observations == {date(2026, 6, 1): 4.0, date(2026, 7, 1): 4.2}  # June revised 4.1 -> 4.0
    # on release day before 17:00 ET the previous vintage is the latest visible
    early = src.get("AAA", datetime(2026, 8, 7, 12, tzinfo=NY)).macro["UNRATE"]
    assert early.observations == {date(2026, 6, 1): 4.1}
