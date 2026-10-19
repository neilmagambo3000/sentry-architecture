from datetime import date, datetime, timedelta, timezone

import pytest

from sentry.config import CostModel
from sentry.execution.fake_broker import FakeBroker
from sentry.execution.orders import submit_orders, to_orders
from sentry.execution.reconcile import reconcile
from sentry.execution.tradelog import clock_start, closed_trades_count, record_fills, trading_days
from sentry.types import AccountState, Decision, Position

UTC = timezone.utc
T0 = datetime(2026, 10, 1, 13, 35, tzinfo=UTC)
H = "ab" * 32
ZERO_COST = CostModel(0.0, 0.0)


def dec(sym, w):
    return Decision(sym, T0, (w > 0) - (w < 0), w, None, 0.5, {}, "ph", "v1.0.0", H)


def acct(positions, equity=100.0):
    return AccountState(T0, equity, equity, positions, (), equity, equity, True)


def test_to_orders_notional_close_flip_and_min():
    a = acct({"B": Position("B", 0.5, 10.0, 5.0), "C": Position("C", -0.3, 10.0, -3.0), "D": Position("D", 0.2, 10.0, 2.0)})
    prices = {"A": 20.0, "B": 10.0, "C": 10.0, "D": 10.0, "E": 10.0}
    orders = to_orders([dec("A", 0.05), dec("B", 0.0), dec("C", 0.04), dec("D", 0.025), dec("E", 0.005)], a, prices,
                       "v1.0.0", H, T0)
    by = {o.symbol: o for o in orders}
    assert set(by) == {"A", "B", "C"}  # D delta 0.50 and E delta 0.50 are under min_notional
    assert by["A"].side == "buy" and by["A"].notional == 5.0 and by["A"].qty is None
    assert by["B"].side == "sell" and by["B"].qty == 0.5 and by["B"].notional is None
    assert by["C"].side == "buy" and by["C"].qty == 0.3  # flip closes only
    assert by["A"].client_order_id == f"sentry-{H[:8]}-A-20261001-buy"
    assert all(o.strategy_hash == H and o.strategy_version == "v1.0.0" and o.time_in_force == "day" for o in orders)


def test_submit_orders_idempotent_and_dry_run(db):
    b = FakeBroker(100.0, ZERO_COST, [], {"A": 10.0})
    orders = to_orders([dec("A", 0.05)], acct({}), {"A": 10.0}, "v1.0.0", H, T0)
    assert submit_orders(b, orders, db, dry_run=True) == [(orders[0].client_order_id, "dry_run")]
    assert b.open_orders() == []
    row = db.execute("select dry_run, status, strategy_hash from orders").fetchone()
    assert tuple(row) == (1, "dry_run", H)
    assert submit_orders(b, orders, db, dry_run=False)[0][1] == "skipped_duplicate"
    assert b.open_orders() == []


def test_submit_orders_skips_open_and_logged(db):
    b = FakeBroker(100.0, ZERO_COST, [], {"A": 10.0})
    orders = to_orders([dec("A", 0.05)], acct({}), {"A": 10.0}, "v1.0.0", H, T0)
    assert submit_orders(b, orders, db, dry_run=False)[0][1] == "submitted"
    assert len(b.open_orders()) == 1
    assert submit_orders(b, orders, db, dry_run=False)[0][1] == "skipped_duplicate"
    db.execute("delete from orders")
    assert submit_orders(b, orders, db, dry_run=False)[0][1] == "skipped_duplicate"  # still open at broker
    assert len(b.open_orders()) == 1


def test_submit_error_recorded(db):
    class Boom(FakeBroker):
        def submit(self, order):
            raise RuntimeError("rejected")

    b = Boom(100.0, ZERO_COST, [], {"A": 10.0})
    orders = to_orders([dec("A", 0.05)], acct({}), {"A": 10.0}, "v1.0.0", H, T0)
    assert submit_orders(b, orders, db, dry_run=False)[0][1] == "error"
    assert db.execute("select status from orders").fetchone()[0] == "error"


def test_fake_broker_cancel_calendar_prices():
    days = [date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 5)]
    b = FakeBroker(100.0, ZERO_COST, days, {"A": 10.0})
    assert b.calendar(date(2026, 10, 2), date(2026, 10, 5)) == days[1:]
    assert b.latest_prices(["A", "Z"]) == {"A": 10.0}
    oid = b.submit(to_orders([dec("A", 0.05)], acct({}), {"A": 10.0}, "v", H, T0)[0])
    b.cancel(oid)
    assert b.open_orders() == [] and b.fill_open_orders(T0) == []
    assert b.is_paper is True


def test_reconcile_builds_account_and_flags_diffs(db):
    b = FakeBroker(100.0, ZERO_COST, [], {"A": 10.0})
    b.mark_close(T0 - timedelta(days=2))
    b.submit(to_orders([dec("A", 0.1)], acct({}), {"A": 10.0}, "v1.0.0", H, T0)[0])
    fills = b.fill_open_orders(T0, {"A": 10.0})
    record_fills(db, fills, "v1.0.0", H)
    b.set_prices({"A": 12.0})
    b.mark_close(T0 + timedelta(hours=3))  # equity 102 is the new peak and last close
    b.set_prices({"A": 11.0})
    now = T0 + timedelta(days=1)
    state, diffs = reconcile(b, db, "v1.0.0", now, expected_positions={"A": 1.0, "B": 2.0})
    assert state.equity == pytest.approx(101.0) and state.peak_equity == pytest.approx(102.0)
    assert state.day_start_equity == pytest.approx(102.0) and state.is_paper
    assert state.positions["A"].qty == pytest.approx(1.0) and state.positions["A"].opened_at == T0
    assert diffs == [{"symbol": "B", "expected": 2.0, "actual": 0.0, "reconcile_diff": -2.0}]
    assert db.execute("select count(*) from positions_snapshot").fetchone()[0] == 1
    assert trading_days(db, "v1.0.0") == 1
    _, diffs = reconcile(b, db, "v1.0.0", now, expected_positions={"A": 1.0})
    assert diffs == []


def test_tradelog_closed_trades_partial_and_flip(db):
    b = FakeBroker(100.0, ZERO_COST, [], {"A": 10.0})

    def trade(side, qty, price, t):
        from sentry.types import Order

        b.submit(Order("A", side, None, qty, f"c{t}", "v1.0.0", H))
        record_fills(db, b.fill_open_orders(T0 + timedelta(hours=t), {"A": price}), "v1.0.0", H)

    trade("buy", 2, 10.0, 1)
    trade("buy", 2, 12.0, 2)  # avg 11
    trade("sell", 1, 13.0, 3)  # partial: +2
    assert closed_trades_count(db, "v1.0.0") == 0
    trade("sell", 5, 9.0, 4)  # closes 3 at 9 (-6), flips to short 2 at 9
    trade("buy", 2, 8.0, 5)  # closes short: +2
    rows = db.execute("select side, qty, entry_price, exit_price, pnl from closed_trades order by closed_at").fetchall()
    assert [tuple(r) for r in rows] == [(1, 4.0, 11.0, pytest.approx(10.0), pytest.approx(-4.0)),
                                       (-1, 2.0, 9.0, 8.0, pytest.approx(2.0))]
    assert closed_trades_count(db, "v1.0.0") == 2 and closed_trades_count(db, "v9") == 0
    assert b.equity() == pytest.approx(100.0 - 4.0 + 2.0)


def test_version_clock_ignores_dry_run_burn_in(db):
    b = FakeBroker(100.0, ZERO_COST, [], {"A": 10.0})
    for day in range(1, 11):  # ten dry-run burn-in days: reconcile every day, orders logged not sent
        t = T0.replace(day=day)
        reconcile(b, db, "v1.0.0", t)
        submit_orders(b, to_orders([dec("A", 0.05)], acct({}), {"A": 10.0}, "v1.0.0", H, t), db, dry_run=True)
    assert clock_start(db, "v1.0.0") is None and trading_days(db, "v1.0.0") == 0
    t = T0.replace(day=12)  # first non-dry-run order: the clock starts here
    reconcile(b, db, "v1.0.0", t)
    submit_orders(b, to_orders([dec("A", 0.05)], acct({}), {"A": 10.0}, "v1.0.0", H, t), db, dry_run=False)
    db.execute("UPDATE orders SET submitted_at = ? WHERE dry_run = 0", (t.isoformat(),))
    assert clock_start(db, "v1.0.0") == "2026-10-12" and trading_days(db, "v1.0.0") == 1
    reconcile(b, db, "v1.0.0", T0.replace(day=13))
    assert trading_days(db, "v1.0.0") == 2 and trading_days(db, "v2.0.0") == 0
