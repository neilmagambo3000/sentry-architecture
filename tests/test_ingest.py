"""make_ingest with fake loaders and an injected clock: no network, no from_env."""
from datetime import date, datetime, timezone

from sentry.data.store import connect
from sentry.live.ingest import make_ingest

UTC = timezone.utc
AS_OF = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)  # 08:00 NY, Wed 2026-09-30
ARRIVAL = datetime(2026, 9, 30, 12, 0, 5, tzinfo=UTC)


class Fake:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def _do(self, *a, **k):
        self.calls.append((a, k))
        if self.fail:
            raise RuntimeError("vendor down")
        return 3

    load = _do


def build(**fail):
    f = {n: Fake(n in fail) for n in ("bars", "news", "edgar", "alfred")}
    fn = make_ingest(f["bars"], f["news"], f["edgar"], f["alfred"], ["AAA", "BBB"], {"cpi": "CPIAUCSL"},
                     lambda: ARRIVAL, news_window_days=7)
    return f, fn


def test_ranges_with_empty_store_and_clock_arrival():
    conn = connect(":memory:")
    f, fn = build()
    rep = fn(conn, AS_OF)
    assert all(v["status"] == "ok" for v in rep.values())
    (a, _), = f["bars"].calls[:1]
    assert a[1] == ["AAA"] and a[2] == date(2024, 8, 31) and a[3] == date(2026, 9, 29)
    (na, nk), = f["news"].calls
    assert na[2] == datetime(2026, 9, 23, 12, tzinfo=UTC) and nk["arrival_time"] == ARRIVAL
    assert f["alfred"].calls[0][0][1:] == ("CPIAUCSL", date(2024, 8, 31))


def test_starts_from_last_stored_rows():
    conn = connect(":memory:")
    conn.execute("INSERT INTO bars (symbol, date) VALUES ('AAA', '2026-09-25')")
    conn.execute("INSERT INTO articles (id, first_seen) VALUES ('x', '2026-09-29T10:00:00+00:00')")
    conn.commit()
    f, fn = build()
    fn(conn, AS_OF)
    assert f["bars"].calls[0][0][2] == date(2026, 9, 26)
    assert f["news"].calls[0][0][2] == datetime(2026, 9, 29, 10, tzinfo=UTC)


def test_edgar_once_per_day():
    conn = connect(":memory:")
    f, fn = build()
    fn(conn, AS_OF)
    fn(conn, AS_OF)
    assert len(f["edgar"].calls) == 1 and len(f["news"].calls) == 2


def test_failing_loader_is_reported_and_others_run():
    conn = connect(":memory:")
    f, fn = build(news=True)
    rep = fn(conn, AS_OF)
    assert rep["news"]["status"] == "error" and "vendor down" in rep["news"]["error"]
    assert rep["bars"]["status"] == rep["edgar"]["status"] == rep["alfred"]["status"] == "ok"
    assert f["edgar"].calls and f["alfred"].calls
