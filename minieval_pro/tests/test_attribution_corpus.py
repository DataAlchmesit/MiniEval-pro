"""
Runs the attribution regression corpus (attribution_corpus.py) against the
real AttributionHook — real DeBERTa + MiniLM models, no fakes.

Run with:
    pytest minieval_pro/tests/test_attribution_corpus.py -v
"""

import pytest

from minieval_pro.gate import MemoryGate
from minieval_pro.hooks.attribution_hook import AttributionHook
from minieval_pro.tests.attribution_corpus import CORPUS


@pytest.fixture(scope="module")
def hook():
    return AttributionHook(gate=MemoryGate(quiet=True))


@pytest.mark.parametrize("case", CORPUS, ids=[c.id for c in CORPUS])
def test_corpus_case(hook, case):
    result = hook.check(candidate=case.candidate, source_text=case.source_text)

    print(f"\n  [{case.category}] verdict={result.verdict}  "
          f"reason={result.reason!r}")

    assert result.verdict == case.expected, (
        f"{case.id} ({case.category}): expected {case.expected}, got "
        f"{result.verdict}. {case.note}"
    )


def test_corpus_covers_required_categories():
    """
    chenhz01 asked specifically for sibling/family possession and
    reported speech coverage (temporal shifts deliberately excluded —
    see attribution_corpus.py docstring for why). This confirms the
    corpus actually has cases in both required categories, so an empty
    or misnamed category can't silently pass by having zero cases.
    """
    categories = {c.category for c in CORPUS}
    assert "sibling_family_possession" in categories
    assert "reported_speech" in categories

    family_count = sum(1 for c in CORPUS if c.category == "sibling_family_possession")
    reported_count = sum(1 for c in CORPUS if c.category == "reported_speech")
    assert family_count >= 3, "Too few family-possession cases to call this a corpus"
    assert reported_count >= 3, "Too few reported-speech cases to call this a corpus"