"""scripts/backfill.py core with fake loaders only: never from_env, no network."""
import importlib.util
import sys
from datetime import datetime, timezone

from sentry.config import ROOT
from sentry.data.store import connect

spec = importlib.util.spec_from_file_location("backfill", ROOT / "scripts" / "backfill.py")
bf = importlib.util.module_from_spec(spec)
sys.modules["backfill"] = bf
spec.loader.exec_module(bf)

AS_OF = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


class Fake:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def load(self, *a, **k):
        self.calls.append(a)
        if self.fail:
            raise RuntimeError("vendor down")
        return 2


def test_backfill_loads_every_source_and_reports():
    f = [Fake(), Fake(), Fake(), Fake()]
    rep = bf.backfill(connect(":memory:"), f, ["AAA", "BBB"], {"cpi": "CPIAUCSL"}, AS_OF, news_days=30,
                      clock=lambda: AS_OF)
    assert rep == {n: {"status": "ok", "count": c} for n, c in (("bars", 4), ("news", 2), ("edgar", 2), ("alfred", 2))}
    assert f[1].calls[0][2] == datetime(2026, 8, 31, 12, tzinfo=timezone.utc)  # 30-day news window


def test_one_vendor_failure_is_reported_not_raised():
    f = [Fake(), Fake(fail=True), Fake(), Fake()]
    rep = bf.backfill(connect(":memory:"), f, ["AAA"], {}, AS_OF, clock=lambda: AS_OF)
    assert rep["news"]["status"] == "error" and rep["bars"]["status"] == "ok"
