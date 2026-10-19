"""Deterministic article digest and daily digest (no LLM, zero cost)."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable

from sentry.agents.tagger import Tagger
from sentry.types import Article

JACCARD_DUP = 0.6
MAX_BULLETS = 12


def _shingles(headline: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", headline.lower())
    if len(words) < 2:
        return {tuple(words)}
    return {tuple(words[i:i + 2]) for i in range(len(words) - 1)}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def relative_time(as_of: datetime, t: datetime) -> str:
    mins = max(0, int((as_of - t).total_seconds() // 60))
    if mins < 60:
        return f"{mins}m ago"
    if mins < 48 * 60:
        return f"{mins // 60}h ago"
    return f"{mins // 1440}d ago"


def article_digest(articles: Iterable[Article], symbol: str, as_of: datetime, tagger: Tagger,
                   max_items: int = MAX_BULLETS) -> tuple[list[str], list[str]]:
    """Returns (bullet lines, ids rendered). Ranked by relevance then recency; near-duplicates collapsed."""
    tagged = [(a, *tagger.tag(a, symbol)) for a in articles]
    tagged = [t for t in tagged if t[1] > 0]
    tagged.sort(key=lambda t: (-t[1], -t[0].first_seen.timestamp(), t[0].id))
    kept: list[tuple[Article, float, object, set]] = []
    for a, rel, sent in tagged:
        sh = _shingles(a.headline)
        if any(_jaccard(sh, k[3]) >= JACCARD_DUP for k in kept):
            continue
        kept.append((a, rel, sent, sh))
        if len(kept) == max_items:
            break
    lines = []
    for a, _, sent, _ in kept:
        line = f"[{a.id}] {a.headline} ({a.source}, {relative_time(as_of, a.first_seen)})"
        if sent is not None:
            line += f" sentiment={sent:+.2f}"
        lines.append(line)
    return lines, [k[0].id for k in kept]


def daily_digest(as_of: datetime, macro_regime: str, universe_summary: str) -> str:
    """Shared across all symbols for one as_of (second cache breakpoint). No absolute date or run id is
    rendered here: only the weekday, so the cached prefix carries no timestamp."""
    return (f"Daily market context\nWeekday: {as_of.strftime('%A')}\nMacro regime: {macro_regime}\n"
            f"Universe summary:\n{universe_summary.strip()}\n")
