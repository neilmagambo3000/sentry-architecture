"""Usage -> USD and the daily LLM cost ceiling (table cost_daily)."""
from __future__ import annotations

import sqlite3
from datetime import date
from typing import Mapping

from sentry.config import load_settings


class CostCeilingError(RuntimeError):
    pass


def usage_usd(usage, prices: Mapping[str, float]) -> float:
    """Anthropic's input_tokens excludes cache reads/writes, so the four meters are priced separately."""
    return (getattr(usage, "input_tokens", 0) * prices["input"]
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * prices["cache_write"]
            + (getattr(usage, "cache_read_input_tokens", 0) or 0) * prices["cache_read"]
            + getattr(usage, "output_tokens", 0) * prices["output"]) / 1e6


def estimate_usd(input_chars: int, max_tokens: int, prices: Mapping[str, float]) -> float:
    """Conservative: chars/3 tokens, all priced as uncached input, plus a full max_tokens of output."""
    return ((input_chars / 3) * prices["input"] + max_tokens * prices["output"]) / 1e6


class CostBudget:
    def __init__(self, conn: sqlite3.Connection, daily_usd: float, day: date,
                 prices: Mapping[str, float] | None = None):
        self.conn, self.daily_usd, self.day = conn, daily_usd, day
        self.prices = dict(prices) if prices is not None else dict(load_settings().models.price_per_mtok)

    def spent_today(self) -> float:
        row = self.conn.execute("SELECT usd FROM cost_daily WHERE date = ?", (self.day.isoformat(),)).fetchone()
        return float(row[0]) if row else 0.0

    def would_exceed(self, estimate: float) -> bool:
        return self.spent_today() + estimate > self.daily_usd

    def record(self, usd: float) -> None:
        if usd <= 0:
            return
        self.conn.execute(
            "INSERT INTO cost_daily (date, usd) VALUES (?, ?) ON CONFLICT(date) DO UPDATE SET usd = usd + excluded.usd",
            (self.day.isoformat(), usd))
        self.conn.commit()
