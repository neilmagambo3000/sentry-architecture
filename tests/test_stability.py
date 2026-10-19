import math
from types import SimpleNamespace as V

import pytest

from sentry.layer_a.validation.stability import confirm_spend, estimate_usd, main, stability

PRICES = {"input": 2.0, "output": 10.0, "cache_write": 2.5, "cache_read": 0.2}


def test_stability_flip_rate_and_conf_std():
    draws = {
        "steady": [V(direction=1, confidence=0.6)] * 5,
        "flippy": [V(direction=1, confidence=0.4), V(direction=-1, confidence=0.6), V(direction=1, confidence=0.5),
                   V(direction=0, confidence=0.5), V(direction=1, confidence=0.5)],
        "dead": [None] * 5,  # failed calls are excluded
    }
    calls = []

    def fake(bundle, i):
        calls.append((bundle, i))
        return draws[bundle][i]

    r = stability(list(draws), draws=5, call_fn=fake)
    assert len(calls) == 15
    assert r["n"] == 2
    assert r["flip_rate"] == pytest.approx(2 / 10)
    assert r["conf_std"] == pytest.approx((0 + 0.0632455532) / 2, rel=1e-6)


def test_stability_empty_and_requires_call_fn():
    assert math.isnan(stability([], call_fn=lambda b, i: None)["flip_rate"])
    with pytest.raises(ValueError):
        stability(["b"])


def test_estimate_usd():
    assert estimate_usd(200, 5, 7000, 500, PRICES) == pytest.approx(1000 * (7000 * 2 + 500 * 10) / 1e6)


def test_confirm_requires_exact_phrase(capsys):
    assert confirm_spend(19.0, input_fn=lambda _: "SPEND 19.00")
    assert not confirm_spend(19.0, input_fn=lambda _: "yes")


def test_main_aborts_without_consent_and_never_spends(capsys):
    assert main(["--rows", "2"], input_fn=lambda _: "no") == 1
    assert "Aborted" in capsys.readouterr().out
    assert main(["--rows", "2", "--draws", "1"], input_fn=lambda _: "SPEND 0.04") == 2
