from datetime import date
from types import SimpleNamespace

import pytest

from sentry.llm.cost import CostBudget, estimate_usd, usage_usd

PRICES = {"input": 2.0, "output": 10.0, "cache_write": 2.5, "cache_read": 0.2}


def test_usage_usd_prices_four_meters():
    u = SimpleNamespace(input_tokens=1_000_000, cache_creation_input_tokens=1_000_000,
                        cache_read_input_tokens=1_000_000, output_tokens=1_000_000)
    assert usage_usd(u, PRICES) == pytest.approx(2.0 + 2.5 + 0.2 + 10.0)


def test_usage_usd_tolerates_none_cache_meters():
    u = SimpleNamespace(input_tokens=1000, cache_creation_input_tokens=None, cache_read_input_tokens=None,
                        output_tokens=100)
    assert usage_usd(u, PRICES) == pytest.approx((1000 * 2 + 100 * 10) / 1e6)


def test_estimate_is_conservative():
    # 3000 chars -> 1000 tokens uncached input + full 1024 output
    assert estimate_usd(3000, 1024, PRICES) == pytest.approx((1000 * 2 + 1024 * 10) / 1e6)


def test_budget_records_per_day(db, settings):
    b = CostBudget(db, 1.0, date(2026, 9, 30), prices=PRICES)
    assert b.spent_today() == 0 and not b.would_exceed(0.5)
    b.record(0.6)
    b.record(0.3)
    b.record(0.0)
    assert b.spent_today() == pytest.approx(0.9)
    assert b.would_exceed(0.2) and not b.would_exceed(0.1)
    assert CostBudget(db, 1.0, date(2026, 10, 1), prices=PRICES).spent_today() == 0


def test_budget_default_prices_from_models_yaml(db, settings):
    assert CostBudget(db, 1.0, date(2026, 9, 30)).prices == dict(settings.models.price_per_mtok)
