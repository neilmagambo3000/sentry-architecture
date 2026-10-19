import math
import random
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from sentry.config import ROOT
from sentry.risk.gate import RiskGate
from sentry.risk.limits import load_limits
from sentry.types import AccountState, Decision, EventCalendar, Position

L = load_limits(ROOT / "config")
NOW = datetime(2026, 10, 1, 12, 30, tzinfo=timezone.utc)
SNAP = NOW - timedelta(minutes=30)


def dec(sym, w):
    return Decision(sym, NOW, (w > 0) - (w < 0), w, None, 0.5, {}, "ph", "v1.0.0", "h" * 64)


def acct(positions=None, equity=100.0, peak=100.0, day_start=100.0, paper=True):
    ps = {s: Position(s, w * equity / 10.0, 10.0, w * equity) for s, w in (positions or {}).items()}
    return AccountState(NOW, equity, equity, ps, (), day_start, peak, paper)


@pytest.fixture
def gate(tmp_path):
    return RiskGate(L, tmp_path / "KILL")


def run(gate, decisions, account=None, cal=None, snap=SNAP, cost=0.0):
    return gate.check(decisions, account or acct(), cal or EventCalendar(), NOW, snap, cost)


def weights(res):
    return {d.symbol: d.target_weight for d in res.approved}


def test_clean_pass_through(gate):
    res = run(gate, [dec("A", 0.05), dec("B", 0.04)])
    assert weights(res) == {"A": 0.05, "B": 0.04} and not res.halt_all and res.reasons == ()


def test_long_only_reduces_shorts_to_zero(gate):
    assert L.long_only is True  # config/risk.yaml
    res = run(gate, [dec("A", 0.05), dec("B", -0.04)])
    assert weights(res) == {"A": 0.05, "B": 0.0} and res.approved[1].direction == 0
    assert any("long_only" in r for r in res.reasons)
    res = run(gate, [dec("C", -0.05)], acct({"C": -0.03}))  # a legacy short is closed, never increased
    assert weights(res) == {"C": 0.0}
    loose = RiskGate(replace(L, long_only=False), gate.kill_switch_path)
    assert weights(run(loose, [dec("B", -0.04)])) == {"B": -0.04}


def test_close_to_close_daily_drawdown_blocks_entries_today(gate):
    """At 08:00 equity ~= yesterday's close, so the drop must be read from broker history."""
    def hist(*eqs):
        return tuple((NOW - timedelta(days=len(eqs) - i), e) for i, e in enumerate(eqs))

    a = replace(acct({"B": 0.05}, equity=94.9), equity_history=hist(100.0, 94.9), day_start_equity=94.9)
    res = run(gate, [dec("A", 0.05), dec("B", 0.0)], a)
    assert res.no_new_entries and not res.halt_all and not res.flatten
    assert weights(res) == {"B": 0.0}  # entry dropped, exit still allowed
    assert any("close-to-close" in r for r in res.reasons)
    a = replace(a, equity_history=hist(100.0, 95.1), equity=95.1, day_start_equity=95.1)
    assert not run(gate, [dec("A", 0.05)], a).no_new_entries
    today = ((NOW.replace(hour=1), 90.0),)  # a point dated today is not a completed day
    a = replace(a, equity_history=hist(100.0, 99.0) + today)
    assert not run(gate, [dec("A", 0.05)], a).no_new_entries


def test_not_paper_halts(gate):
    res = run(gate, [dec("A", 0.05)], acct(paper=False))
    assert res.halt_all and not res.flatten and res.approved == () and "paper" in res.reasons[0]


def test_kill_switch_flattens(gate, tmp_path):
    (tmp_path / "KILL").touch()
    res = run(gate, [dec("A", 0.05)], acct({"B": 0.05, "C": -0.03}))
    assert res.halt_all and res.flatten and weights(res) == {"B": 0.0, "C": 0.0}
    assert "kill switch" in res.reasons[0]


def test_stale_snapshot_halts_without_flatten(gate):
    res = run(gate, [dec("A", 0.05)], acct({"B": 0.05}), snap=NOW - timedelta(hours=24.5))
    assert res.halt_all and not res.flatten and res.approved == ()
    assert run(gate, [dec("A", 0.05)], snap=NOW - timedelta(hours=23.9)).approved
    assert run(gate, [dec("A", 0.05)], snap=None).halt_all


def test_llm_cost_blocks_new_entries_only(gate):
    a = acct({"B": 0.08, "C": 0.05, "D": 0.05})
    res = run(gate, [dec("A", 0.05), dec("B", 0.04), dec("C", 0.09), dec("D", -0.05)], a, cost=5.0)
    assert res.no_new_entries and not res.halt_all
    assert weights(res) == {"B": 0.04, "D": 0.0}  # A entry dropped, C increase dropped, D flip -> close
    assert len(res.reasons) == 4
    assert not run(gate, [dec("A", 0.05)], cost=4.99).no_new_entries


def test_total_drawdown_flattens(gate):
    res = run(gate, [dec("A", 0.05)], acct({"B": 0.05}, equity=79.99, peak=100.0, day_start=79.99))
    assert res.halt_all and res.flatten and weights(res) == {"B": 0.0}
    res = run(gate, [dec("A", 0.05)], acct({"B": 0.05}, equity=80.01, peak=100.0, day_start=80.01))
    assert not res.halt_all


def test_daily_drawdown_halts_for_the_day(gate):
    res = run(gate, [dec("A", 0.05)], acct({"B": 0.05}, equity=95.0, peak=100.0, day_start=100.0))
    assert res.halt_all and res.no_new_entries and not res.flatten and res.approved == ()
    assert "daily drawdown" in res.reasons[0]


def test_earnings_blackout(gate):
    cal = EventCalendar({"A": 2, "B": 0, "C": 1, "D": 3})
    a = acct({"B": 0.05, "C": 0.05})
    res = run(gate, [dec("A", 0.05), dec("B", 0.03), dec("C", -0.05), dec("D", 0.05)], a, cal)
    assert weights(res) == {"B": 0.03, "C": 0.0, "D": 0.05}


def test_clamp_and_gross_scaling(gate):
    res = run(gate, [dec(s, 0.3) for s in "ABCDEFGH"])
    w = weights(res)
    assert all(v == pytest.approx(0.8 / 8) for v in w.values())  # clamped to 0.20, then 1.6 gross scaled to 0.8
    assert any("clamped" in r for r in res.reasons) and any("gross" in r for r in res.reasons)


def test_non_finite_target_dropped(gate):
    assert weights(run(gate, [dec("A", float("nan"))])) == {}


def test_property_gate_only_reduces(tmp_path):
    rng = random.Random(20260930)
    kill = tmp_path / "KILL"
    gate = RiskGate(L, kill)
    syms = list("ABCDEFGHIJ")
    for _ in range(4000):
        if rng.random() < 0.05:
            kill.touch()
        elif kill.exists():
            kill.unlink()
        equity = rng.uniform(50, 150)
        peak = equity * rng.uniform(1.0, 1.12)
        day_start = equity * rng.uniform(0.97, 1.03)
        held = {s: rng.uniform(-0.15, 0.15) for s in rng.sample(syms, rng.randint(0, 6))}
        account = acct(held, equity, peak, day_start, paper=rng.random() > 0.03)
        cal = EventCalendar({s: rng.choice([None, 0, 1, 2, 5]) for s in syms})
        decisions = [
            dec(s, rng.choice([0.0, rng.uniform(-0.3, 0.3), rng.uniform(-0.05, 0.05)]))
            for s in rng.sample(syms, rng.randint(0, 10))
        ]
        res = gate.check(decisions, account, cal, NOW, NOW - timedelta(hours=rng.uniform(0, 24)), rng.uniform(0, 15))
        inputs = {d.symbol: d.target_weight for d in decisions}
        cur = {s: p.market_value / equity for s, p in account.positions.items()}
        assert len({d.symbol for d in res.approved}) == len(res.approved)
        if res.halt_all and not res.flatten:
            assert res.approved == ()
        if res.flatten:
            assert all(d.target_weight == 0.0 for d in res.approved)
        gross = 0.0
        for d in res.approved:
            a = d.target_weight
            assert math.isfinite(a)
            if d.symbol in inputs:
                assert abs(a) <= abs(inputs[d.symbol]) + 1e-12
                assert a * inputs[d.symbol] >= 0  # never flips versus the input
            else:
                assert a == 0.0
            assert abs(a) <= L.max_weight + 1e-12
            if res.no_new_entries:
                c = cur.get(d.symbol, 0.0)
                assert abs(a) <= abs(c) + 1e-12 and a * c >= 0
            gross += abs(a)
        assert gross <= L.max_gross + 1e-9
