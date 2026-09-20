"""
Tests the sentence-narrowing fix (attribution + relatedness guards)
against the REAL DeBERTa + MiniLM models — no fakes.

Confirms whether narrowing to the most-relevant sentence fixes the
multi-fact contamination problem found via test_mem0_adapter.py: a
third-party mention anywhere in a multi-topic source previously caused
unrelated facts extracted from the same source to be wrongly flagged.

Run with:
    pytest minieval_pro/tests/test_attribution_narrowing.py -v
"""

from dataclasses import dataclass

import pytest

from minieval_pro.gate import MemoryGate, Policy


SOURCE_TEXT = (
    "I moved to Chennai last week. I had a great salad for lunch. "
    "My brother is a lawyer and he helped me with the contract."
)


@dataclass
class NarrowingCase:
    label: str
    fact: str
    expected: str
    note: str = ""


CASES = [
    NarrowingCase(
        label="chennai_not_contaminated_by_brother_sentence",
        fact="The user lives in Chennai.",
        expected="STORE",
        note=(
            "Unrelated to the brother sentence. Before narrowing, the "
            "attribution guard scanned the WHOLE source and wrongly "
            "flagged this because 'my brother' appeared anywhere in it."
        ),
    ),
    NarrowingCase(
        label="peanuts_rejected_via_semantic_relatedness",
        fact="The user loves eating peanuts.",
        expected="REJECT",
        note=(
            "Contradicts its own (salad) sentence. This is the case that "
            "needed the REAL MiniLM model to confirm — 'salad for lunch' "
            "and 'eating peanuts' share no literal words, only semantic "
            "similarity (both food). A keyword-overlap fake scorer could "
            "not validate this case; only a real embedding model can."
        ),
    ),
    NarrowingCase(
        label="lawyer_misattribution_still_caught",
        fact="The user is a lawyer.",
        expected="REVIEW",
        note="The actual misattribution — narrowing must not break this.",
    ),
]


@pytest.fixture(scope="module")
def gate():
    return MemoryGate(policy=Policy(), quiet=True)


@pytest.mark.parametrize("case", CASES, ids=[c.label for c in CASES])
def test_narrowing_case(gate, case):
    decision = gate.check(source=SOURCE_TEXT, fact=case.fact)

    print(
        f"\n  faithfulness={decision.faithfulness:.2f} ({decision.label})  "
        f"relatedness={decision.relatedness}"
    )

    assert decision.verdict == case.expected, (
        f"{case.label}: expected {case.expected}, got {decision.verdict}. "
        f"{case.note}"
    )