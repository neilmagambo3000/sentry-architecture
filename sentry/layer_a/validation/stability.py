"""Pre-freeze stability check: re-call fixed bundles several times with the cache bypassed.

Lives under layer_a/, so it must not import sentry.llm / sentry.agents / anthropic (§3.4). The paid
caller is injected as call_fn(bundle, draw) -> view-like object with .direction/.confidence, or None.
It costs money and is NOT run in tests: the entry point prints an estimate and requires typed consent.
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter
from typing import Callable, Iterable, Mapping, Optional

from sentry.config import load_settings


def stability(bundles: Iterable[object], draws: int = 5,
              call_fn: Optional[Callable[[object, int], object]] = None) -> dict:
    """flip_rate = share of valid draws whose direction differs from that bundle's modal direction.
    conf_std = mean over bundles of the population std of confidence. n = bundles with >= 2 valid draws."""
    if call_fn is None:
        raise ValueError("call_fn is required (it must bypass the response cache)")
    flips = total = 0
    stds = []
    for b in bundles:
        views = [v for v in (call_fn(b, i) for i in range(draws)) if v is not None]
        if len(views) < 2:
            continue
        mode = Counter(v.direction for v in views).most_common(1)[0][0]
        flips += sum(v.direction != mode for v in views)
        total += len(views)
        stds.append(statistics.pstdev(v.confidence for v in views))
    return {"flip_rate": flips / total if total else float("nan"),
            "conf_std": statistics.fmean(stds) if stds else float("nan"),
            "n": len(stds)}


def estimate_usd(rows: int, draws: int, input_tokens: int, output_tokens: int, prices: Mapping[str, float]) -> float:
    """Conservative: every input token priced as uncached input (prompt caching would only lower it)."""
    return rows * draws * (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1e6


def confirm_spend(usd: float, input_fn: Callable[[str], str] = input) -> bool:
    phrase = f"SPEND {usd:.2f}"
    print(f"Estimated cost: ${usd:.2f} USD. Type exactly '{phrase}' to continue, anything else aborts.")
    return input_fn("> ").strip() == phrase


def main(argv: list[str] | None = None, input_fn: Callable[[str], str] = input) -> int:
    ap = argparse.ArgumentParser(description="Stability check cost gate (paid; owner approves).")
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--input-tokens", type=int, default=7000)
    ap.add_argument("--output-tokens", type=int, default=500)
    a = ap.parse_args(argv)
    prices = load_settings().models.price_per_mtok
    usd = estimate_usd(a.rows, a.draws, a.input_tokens, a.output_tokens, prices)
    print(f"{a.rows} bundles x {a.draws} draws x ({a.input_tokens} in + {a.output_tokens} out) tokens")
    if not confirm_spend(usd, input_fn):
        print("Aborted, nothing spent.")
        return 1
    # ponytail: bundle loading from the live store and the real call_fn live outside layer_a (boundary
    # rule); wire them in a Layer B script once the four-week dev window exists.
    print("Confirmed, but no bundle source is wired yet: nothing was spent.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
