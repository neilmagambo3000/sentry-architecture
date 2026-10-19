import importlib.util
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sentry.agents.context import build_bundle, system_prompt
from sentry.agents.digest import article_digest, daily_digest
from sentry.agents.tagger import FinBERTTagger, KeywordTagger
from sentry.types import Article, Snapshot

AS_OF = datetime(2026, 9, 30, 12, tzinfo=timezone.utc)


def art(i, headline, symbols=("AAA",), hours=1, source="Reuters"):
    return Article(f"n{i}", symbols, AS_OF - timedelta(hours=hours), None, source, headline, None, "u")


def snap(arts):
    return Snapshot("AAA", AS_OF, pd.DataFrame(), tuple(arts), (), {})


def test_system_prompt_is_cacheable_and_static():
    s = system_prompt("disc")
    assert s.strip()  # public snapshot ships a placeholder prompt
    assert chr(0x2014) not in s  # no em dashes
    assert "2026" not in s


def test_keyword_tagger():
    t = KeywordTagger()
    assert t.tag(art(1, "AAA beats estimates"), "AAA") == (1.0, None)
    assert t.tag(art(2, "Chipmakers rally"), "AAA") == (0.5, None)
    assert t.tag(art(3, "AAAB news", symbols=("AAAB",)), "AAA") == (0.0, None)


def test_digest_dedupes_ranks_and_caps():
    arts = [art(1, "Chipmakers rally on demand", hours=1),
            art(2, "AAA raises full year guidance on strong demand", hours=5),
            art(3, "AAA raises full year guidance on strong demand, shares up", hours=2)]
    arts += [art(10 + i, f"Unrelated sector item number {i} today", hours=10 + i) for i in range(20)]
    lines, ids = article_digest(arts, "AAA", AS_OF, KeywordTagger())
    assert len(lines) == 12 and len(ids) == 12
    assert ids[0] == "n3" and "n2" not in ids  # newer duplicate kept, older collapsed; headline hits rank first
    assert lines[0] == "[n3] AAA raises full year guidance on strong demand, shares up (Reuters, 2h ago)"


def test_daily_digest_has_no_date():
    d = daily_digest(AS_OF, "expansion", "20 names, breadth positive")
    assert "Wednesday" in d and "2026" not in d and "09-30" not in d


def test_bundle_visible_ids_and_layout():
    b = build_bundle(snap([art(1, "AAA wins contract"), art(2, "AAA wins contract")]), "TA text", "FA text",
                     daily_digest(AS_OF, "expansion", "x"), KeywordTagger(), masked=False)
    assert b.visible_ids == frozenset({"n1"})  # duplicate not rendered, so not visible
    assert "[n1]" in b.symbol_context and "[n2]" not in b.symbol_context
    assert AS_OF.isoformat() in b.symbol_context
    assert "2026" not in b.system + b.daily_digest  # nothing time-specific before the last breakpoint


def test_masked_hides_ticker_and_absolute_date():
    arts = [art(1, "AAA and BBB sign deal dated 2026-09-29", symbols=("AAA", "BBB"))]
    b = build_bundle(snap(arts), "AAA technical summary", "AAA last filing 2026-08-01", "digest", KeywordTagger(),
                     masked=True)
    assert "AAA" not in b.symbol_context and "BBB" not in b.symbol_context
    assert "2026" not in b.symbol_context
    assert "SYM0" in b.symbol_context and "SYM1" in b.symbol_context and "D0" in b.symbol_context
    assert b.visible_ids == frozenset({"n1"}) and b.masked


def test_context_hash_changes_with_tagger_and_mask():
    s = snap([art(1, "AAA wins contract")])

    class OtherTagger(KeywordTagger):
        name = "other"

    a = build_bundle(s, "t", "f", "d", KeywordTagger(), masked=False)
    assert a.context_hash == build_bundle(s, "t", "f", "d", KeywordTagger(), masked=False).context_hash
    assert a.context_hash != build_bundle(s, "t", "f", "d", OtherTagger(), masked=False).context_hash
    assert a.context_hash != build_bundle(s, "t", "f", "d", KeywordTagger(), masked=True).context_hash


@pytest.mark.skipif(importlib.util.find_spec("transformers") is not None, reason="transformers installed")
def test_finbert_absent_gives_clear_error():
    with pytest.raises(RuntimeError, match="transformers"):
        FinBERTTagger()
