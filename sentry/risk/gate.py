"""Hard risk gate. Runs after Layer A; no prompt can reach it.

The gate can only reduce, cancel or flatten. Every check that fires appends one reason line
and logs it. Order:
  0. account not paper                       -> halt_all (nothing trades)
  1. kill switch file exists                 -> halt_all + flatten
  2. snapshot older than stale_after_hours   -> halt_all (no flatten)
  3. LLM cost today >= llm_daily_usd         -> no_new_entries
  4. total drawdown from broker peak         -> halt_all + flatten
  5. daily drawdown from day_start_equity    -> no_new_entries + halt_all for this cycle (no flatten)
  6. last completed day's close-to-close equity change (broker history) <= -daily_dd_halt
                                             -> no_new_entries for today (exits still allowed, no flatten).
     Needed because at the 08:00 pre-market cycle equity ~= the previous close, so check 5 almost never fires.
  per decision: long_only (negative target -> 0), no_new_entries filter, earnings blackout, clamp to
  max_weight; then gross scaling.
"""
from __future__ import annotations

import logging
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from sentry.config import RiskLimits
from sentry.types import AccountState, Decision, EventCalendar, GateResult

log = logging.getLogger(__name__)


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


class RiskGate:
    def __init__(self, limits: RiskLimits, kill_switch_path: Path | str):
        self.limits = limits
        self.kill_switch_path = Path(kill_switch_path)

    def check(
        self,
        decisions: Sequence[Decision],
        account: AccountState,
        calendar: EventCalendar,
        now: datetime,
        snapshot_as_of: Optional[datetime],
        llm_cost_today: float,
    ) -> GateResult:
        L = self.limits
        reasons: list[str] = []

        def fire(msg: str) -> None:
            reasons.append(msg)
            log.warning("risk gate: %s", msg)

        def halt(flatten: bool) -> GateResult:
            approved = self._flatten(decisions, account, now, reasons[-1]) if flatten else ()
            return GateResult(approved, halt_all=True, no_new_entries=True, flatten=flatten, reasons=tuple(reasons))

        if not account.is_paper:
            fire("account is not a paper account: halt_all")
            return halt(False)
        if self.kill_switch_path.exists():
            fire(f"kill switch present at {self.kill_switch_path}: halt_all + flatten")
            return halt(True)
        if snapshot_as_of is None:
            fire("no snapshot time: halt_all")
            return halt(False)
        age_h = (now - snapshot_as_of).total_seconds() / 3600.0
        if age_h > L.stale_after_hours:
            fire(f"snapshot is {age_h:.1f}h old > {L.stale_after_hours}h: halt_all")
            return halt(False)

        no_new = False
        if llm_cost_today >= L.llm_daily_usd:
            fire(f"LLM cost today ${llm_cost_today:.2f} >= ${L.llm_daily_usd:.2f}: no_new_entries")
            no_new = True

        eq = account.equity
        if account.peak_equity <= 0 or eq <= 0:
            fire(f"non-positive equity ({eq}) or peak ({account.peak_equity}): halt_all + flatten")
            return halt(True)
        total_dd = 1.0 - eq / account.peak_equity
        if total_dd >= L.total_dd_halt:
            fire(f"total drawdown {total_dd:.2%} from broker peak >= {L.total_dd_halt:.2%}: halt_all + flatten")
            return halt(True)
        if account.day_start_equity > 0:
            day_dd = 1.0 - eq / account.day_start_equity
            if day_dd >= L.daily_dd_halt:
                fire(f"daily drawdown {day_dd:.2%} >= {L.daily_dd_halt:.2%}: no_new_entries, halted for the day")
                return halt(False)
        closes = [e for ts, e in sorted(account.equity_history) if ts.date() < now.date()]
        if len(closes) >= 2 and closes[-2] > 0:
            chg = closes[-1] / closes[-2] - 1.0
            if chg <= -L.daily_dd_halt:
                fire(f"last day close-to-close equity {chg:+.2%} <= -{L.daily_dd_halt:.2%}: no_new_entries today")
                no_new = True

        cur = {s: p.market_value / eq for s, p in account.positions.items() if p.qty != 0}
        kept: list[tuple[Decision, float, list[str]]] = []
        for dec in decisions:
            sym, t, c = dec.symbol, dec.target_weight, cur.get(dec.symbol, 0.0)
            notes: list[str] = []
            if not math.isfinite(t):
                fire(f"{sym}: non-finite target {t}: dropped")
                continue
            if L.long_only and t < 0:
                fire(f"{sym}: long_only: short target {t:+.4f} reduced to 0")
                t = 0.0
                notes.append("long_only")
            reducing = t == 0 or (_sign(t) == _sign(c) and abs(t) <= abs(c))
            flip = c != 0 and t != 0 and _sign(t) != _sign(c)
            blackout_days = calendar.earnings_in_days.get(sym)
            for active, why in (
                (no_new, "no_new_entries"),
                (blackout_days is not None and blackout_days <= L.earnings_blackout_days,
                 f"earnings in {blackout_days}d <= {L.earnings_blackout_days}d blackout"),
            ):
                if not active or reducing:
                    continue
                if flip:
                    fire(f"{sym}: {why}: flip {c:+.4f} -> {t:+.4f} reduced to close only")
                    t, reducing = 0.0, True
                    notes.append(why)
                else:
                    fire(f"{sym}: {why}: entry/increase {c:+.4f} -> {t:+.4f} dropped")
                    t = None
                    break
            if t is None:
                continue
            if abs(t) > L.max_weight:
                fire(f"{sym}: |target| {abs(t):.4f} clamped to max_weight {L.max_weight}")
                t = _sign(t) * L.max_weight
                notes.append("clamped")
            kept.append((dec, t, notes))

        gross = sum(abs(t) for _, t, _ in kept)
        untouched = sum(abs(w) for s, w in cur.items() if s not in {d.symbol for d, _, _ in kept})
        if gross > 0 and gross + untouched > L.max_gross + 1e-12:
            k = max(0.0, L.max_gross - untouched) / gross
            fire(f"gross {gross + untouched:.4f} > max_gross {L.max_gross}: targets scaled by {k:.4f}")
            kept = [(d, t * k, n + ["gross_scaled"]) for d, t, n in kept]

        approved = tuple(
            replace(
                d,
                target_weight=t,
                direction=_sign(t),
                stop=d.stop if t != 0 else None,
                reason={**d.reason, "gate": n} if n else d.reason,
            )
            for d, t, n in kept
        )
        return GateResult(approved, halt_all=False, no_new_entries=no_new, flatten=False, reasons=tuple(reasons))

    @staticmethod
    def _flatten(decisions: Sequence[Decision], account: AccountState, now: datetime, why: str) -> tuple[Decision, ...]:
        """Target 0 for every held position, so to_orders closes each one by qty."""
        tpl = decisions[0] if decisions else None
        return tuple(
            Decision(
                symbol=s, as_of=now, direction=0, target_weight=0.0, stop=None, score=0.0, reason={"gate": why},
                params_hash=tpl.params_hash if tpl else "", strategy_version=tpl.strategy_version if tpl else "",
                strategy_hash=tpl.strategy_hash if tpl else "",
            )
            for s, p in sorted(account.positions.items())
            if p.qty != 0
        )
