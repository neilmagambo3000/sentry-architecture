"""ContextBundle builder.

Cache layout: system (static) | daily_digest (shared per day) | symbol_context.
Nothing time- or run-specific appears before the last breakpoint; as_of only in symbol_context.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from sentry.agents.digest import article_digest
from sentry.agents.tagger import Tagger
from sentry.types import Snapshot

PROMPT_DIR = Path(__file__).parent / "prompts"
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ][0-9:.]+(?:Z|[+-]\d{2}:?\d{2})?)?\b")


@cache
def system_prompt(name: str = "disc") -> str:
    return (PROMPT_DIR / f"{name}.md").read_text()


@dataclass(frozen=True)
class ContextBundle:
    system: str
    daily_digest: str
    symbol_context: str
    visible_ids: frozenset[str]
    context_hash: str
    masked: bool


def _mask(text: str, tokens: dict[str, str]) -> str:
    # ponytail: masks tickers and ISO dates only; company names in headlines still identify the firm.
    # Upgrade path: a symbol -> company-name map from the universe loader.
    for sym, tok in tokens.items():
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(sym)}(?![A-Za-z0-9])", tok, text)
    return _DATE.sub("[date]", text)


def build_bundle(snapshot: Snapshot, ta_text: str, fa_text: str, daily_digest: str, tagger: Tagger,
                 masked: bool) -> ContextBundle:
    lines, ids = article_digest(snapshot.articles, snapshot.symbol, snapshot.as_of, tagger)
    news = "\n".join(f"- {ln}" for ln in lines) if lines else "- (no articles in window)"
    if masked:
        as_of_line = "As of: D0 (decision day; article ages are relative to it)"
        symbol_line = "Symbol: SYM0"
    else:
        as_of_line = f"As of: {snapshot.as_of.isoformat()}"
        symbol_line = f"Symbol: {snapshot.symbol}"
    body = (f"Technical summary:\n{ta_text.strip()}\n\nFundamental summary:\n{fa_text.strip()}\n\n"
            f"News digest ({len(lines)} items; cite only these ids):\n{news}\n")
    if masked:
        visible_articles = [a for a in snapshot.articles if a.id in set(ids)]
        others = sorted({s for a in visible_articles for s in a.symbols} - {snapshot.symbol})
        tokens = {snapshot.symbol: "SYM0", **{s: f"SYM{i + 1}" for i, s in enumerate(others)}}
        body = _mask(body, tokens)
    symbol_context = f"{symbol_line}\n{as_of_line}\n\n{body}\nReturn one AgentView for this symbol."
    system = system_prompt("disc")
    h = hashlib.sha256()
    for part in (system, daily_digest, symbol_context, tagger.name, str(masked)):
        h.update(part.encode() + b"\x1f")
    return ContextBundle(system=system, daily_digest=daily_digest, symbol_context=symbol_context,
                         visible_ids=frozenset(ids), context_hash=h.hexdigest(), masked=masked)
