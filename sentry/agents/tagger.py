"""Article tagger: (relevance, sentiment) per (article, symbol). Deterministic keyword tagger is the v1 default."""
from __future__ import annotations

import re
from typing import Optional, Protocol

from sentry.types import Article


class Tagger(Protocol):
    name: str  # part of the bundle, so changing the tagger changes context_hash

    def tag(self, article: Article, symbol: str) -> tuple[float, Optional[float]]: ...


def _in_headline(symbol: str, headline: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])", headline) is not None


class KeywordTagger:
    name = "keyword-v1"

    def tag(self, article: Article, symbol: str) -> tuple[float, Optional[float]]:
        if symbol not in article.symbols:
            return 0.0, None
        return (1.0 if _in_headline(symbol, article.headline) else 0.5), None


class FinBERTTagger:
    """Optional. transformers/torch are deliberately not in requirements.txt (768 MB container cap)."""

    name = "finbert-prosus-v1"

    def __init__(self):
        try:
            from transformers import pipeline
        except ImportError as e:
            raise RuntimeError("FinBERTTagger needs `transformers` and `torch`, which are not installed "
                               "(not in requirements.txt). Use KeywordTagger or install them.") from e
        self._pipe = pipeline("text-classification", model="ProsusAI/finbert", device=-1, top_k=None)
        self._kw = KeywordTagger()

    def tag(self, article: Article, symbol: str) -> tuple[float, Optional[float]]:
        relevance, _ = self._kw.tag(article, symbol)
        scores = {d["label"].lower(): d["score"] for d in self._pipe(article.headline)[0]}
        return relevance, round(scores.get("positive", 0.0) - scores.get("negative", 0.0), 2)
