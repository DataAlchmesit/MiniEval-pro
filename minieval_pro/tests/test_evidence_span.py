"""
Tests GateDecision.evidence_span against the REAL DeBERTa + MiniLM models.

Run with:
    pytest minieval_pro/tests/test_evidence_span.py -v
"""

from minieval_pro.gate import MemoryGate, Policy


SOURCE_TEXT = (
    "I moved to Chennai last week. I had a great salad for lunch. "
    "My brother is a lawyer and he helped me with the contract."
)


def test_evidence_span_points_at_the_relevant_sentence_chennai():
    gate = MemoryGate(policy=Policy(), quiet=True)
    decision = gate.check(source=SOURCE_TEXT, fact="The user lives in Chennai.")

    assert decision.verdict == "STORE"
    assert decision.evidence_span == "I moved to Chennai last week."


def test_evidence_span_points_at_the_relevant_sentence_lawyer():
    gate = MemoryGate(policy=Policy(), quiet=True)
    decision = gate.check(source=SOURCE_TEXT, fact="The user is a lawyer.")

    assert decision.verdict == "REVIEW"
    assert decision.evidence_span == (
        "My brother is a lawyer and he helped me with the contract."
    )


def test_evidence_span_is_none_for_single_sentence_source():
    """
    Nothing to narrow when the source is already one sentence -- the
    full source is already on the decision, a distinct evidence_span
    would just duplicate it.
    """
    gate = MemoryGate(policy=Policy(), quiet=True)
    decision = gate.check(
        source="I live in Chennai.",
        fact="The user lives in Chennai.",
    )

    assert decision.evidence_span is None