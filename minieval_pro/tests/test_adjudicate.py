"""
Regression suite for gate.adjudicate() — real pytest, not a standalone
script with print statements.

Previous version defined test_existing_timestamp_passthrough() that
computed pass/fail booleans and RETURNED them instead of asserting. Pytest
does not inspect return values — it only fails on a raised exception or a
failed assert — so that function always reported "passed" regardless of
whether the underlying check actually held. Likewise run_suite() and
print_report() were plain functions, not named test_*, so pytest never
collected or ran the 8 real cases at all under `pytest`. Confirmed via a
real run: `python -m pytest` reported "1 passed" while giving zero
visibility into any of the 8 cases.

Fixed by using pytest.mark.parametrize over CASES (each case becomes its
own collected, individually-reportable test) and real `assert` statements
throughout. Run with:

    pytest minieval_pro/tests/test_adjudicate.py -v

-v shows each case by name. Add -s to also see print output if you want
the raw faithfulness scores during a run.
"""

from dataclasses import dataclass

import pytest

from minieval_pro.gate import MemoryGate


@dataclass
class AdjudicateCase:
    label: str
    existing_fact: str
    existing_source: str
    incoming_fact: str
    incoming_source: str
    expected: str            # "ACCEPT", "BLOCK", or "REVIEW"
    note: str = ""


CASES = [

    AdjudicateCase(
        label="case_1_reword",
        existing_fact="The user lives in Bangalore.",
        existing_source="I live in Bangalore.",
        incoming_fact="The user's city is Bangalore.",
        incoming_source="My city is Bangalore.",
        expected="ACCEPT",
        note=(
            "Not a real conflict — just reworded. adjudicate() scores each "
            "side against its own source independently, so a faithful "
            "restatement should ACCEPT cleanly even though nothing "
            "actually changed."
        ),
    ),

    AdjudicateCase(
        label="case_2_genuine_update",
        existing_fact="The user lives in Bangalore.",
        existing_source="I live in Bangalore.",
        incoming_fact="The user lives in Chennai.",
        incoming_source="I moved to Chennai last week.",
        expected="ACCEPT",
        note=(
            "Matches the worked example in README.md. Real move, incoming "
            "is faithful to its own source, so it earns the overwrite."
        ),
    ),

    AdjudicateCase(
        label="case_3_hallucinated_incoming",
        existing_fact="The user is allergic to peanuts.",
        existing_source="I am allergic to peanuts.",
        incoming_fact="The user loves eating peanuts.",
        incoming_source="I had a great salad for lunch.",
        expected="BLOCK",
        note=(
            "Incoming fact is unrelated to its own source, not a genuine "
            "contradiction of it. adjudicate() has no relatedness guard "
            "(unlike check()), so the NLI model forcing this unrelated "
            "pair into 'contradicts' happens to BLOCK correctly here — "
            "right outcome, but for an unverified reason. Documented as a "
            "known limitation, same spirit as check()'s Delhi/Bangalore "
            "case."
        ),
    ),

    AdjudicateCase(
        label="case_4_both_weak",
        existing_fact="The user probably prefers tea over coffee.",
        existing_source="Yeah I guess I might lean towards tea sometimes.",
        incoming_fact="The user prefers coffee over tea.",
        incoming_source="I don't know, coffee's fine I suppose.",
        expected="REVIEW",
        note=(
            "Neither source clearly supports its own fact. Confirms the "
            "safe-default REVIEW path when both sides fall through to "
            "neutral — does not by itself confirm the asymmetric-evidence "
            "gap (see case_4b/4c)."
        ),
    ),

    AdjudicateCase(
        label="case_4b_asymmetric_evidence",
        existing_fact="The user lives in France.",
        existing_source="I live in France.",
        incoming_fact="The user does not live in France.",
        incoming_source="I think I might live in France one day.",
        expected="REVIEW",
        note=(
            "Confirmed via real run: existing=1.00 faithful, "
            "incoming=0.36 faithful — below store_threshold, correctly "
            "falls through to REVIEW. Does not confirm the theorized "
            "asymmetric-evidence gap (adjudicate() never compares "
            "existing_faithfulness to incoming_faithfulness) since "
            "incoming never cleared the threshold to test that path. "
            "Gap remains real by code inspection, unconfirmed in "
            "practice after four attempts (this case plus 4, 4c, and the "
            "original Japanese-fluency attempt)."
        ),
    ),

    AdjudicateCase(
        label="case_4c_asymmetric_evidence_final_attempt",
        existing_fact="The user's favorite color is blue.",
        existing_source="My favorite color has always been blue, ever since I was a kid.",
        incoming_fact="The user's favorite color is green.",
        incoming_source="I painted my room green last year.",
        expected="REVIEW",
        note=(
            "Confirmed via real run: existing=0.90 faithful, "
            "incoming=0.00 neutral. Fourth and final hand-constructed "
            "attempt at the asymmetric-evidence gap — documented as "
            "theoretical-but-unconfirmed in README.md rather than "
            "pursued further by guessing."
        ),
    ),

    AdjudicateCase(
        label="case_5_existing_was_weak",
        existing_fact="The user works as a teacher.",
        existing_source="I think I mentioned I do some tutoring sometimes.",
        incoming_fact="The user works as a software engineer.",
        incoming_source="I've been a software engineer for six years now.",
        expected="ACCEPT",
        note=(
            "Confirmed via real run: existing=0.36 neutral, "
            "incoming=0.98 faithful. A well-evidenced update correctly "
            "replaces a shaky old memory."
        ),
    ),

    AdjudicateCase(
        label="case_6_more_specific_not_contradictory",
        existing_fact="The user lives in Bangalore.",
        existing_source="I live in Bangalore.",
        incoming_fact="The user lives in Koramangala, Bangalore.",
        incoming_source="I live in Koramangala, it's a neighborhood in Bangalore.",
        expected="ACCEPT",
        note=(
            "Confirmed via real run: both scored 0.99 faithful. A "
            "refinement is correctly treated as a clean update, not a "
            "false conflict."
        ),
    ),

]


@pytest.fixture(scope="module")
def gate():
    """One MemoryGate for the whole module — avoids reloading models per test."""
    return MemoryGate(quiet=True)


@pytest.mark.parametrize(
    "case", CASES, ids=[c.label for c in CASES]
)
def test_adjudicate_case(gate, case):
    decision = gate.adjudicate(
        existing_fact=case.existing_fact,
        existing_source=case.existing_source,
        incoming_fact=case.incoming_fact,
        incoming_source=case.incoming_source,
    )

    print(
        f"\n  existing_faithfulness={decision.existing_faithfulness:.2f} "
        f"({decision.existing_label})   "
        f"incoming_faithfulness={decision.incoming_faithfulness:.2f} "
        f"({decision.incoming_label})"
    )

    assert decision.verdict == case.expected, (
        f"{case.label}: expected {case.expected}, got {decision.verdict}. "
        f"{case.note}"
    )


def test_existing_timestamp_omitted_defaults_to_none(gate):
    """existing_timestamp should default to None when the caller doesn't pass it."""
    decision = gate.adjudicate(
        existing_fact="The user lives in Bangalore.",
        existing_source="I live in Bangalore.",
        incoming_fact="The user lives in Chennai.",
        incoming_source="I moved to Chennai last week.",
    )
    assert decision.existing_timestamp is None


def test_existing_timestamp_supplied_passes_through_unchanged(gate):
    """When supplied, existing_timestamp should come back exactly as given."""
    stamp = "2026-03-04T10:15:00+00:00"
    decision = gate.adjudicate(
        existing_fact="The user lives in Bangalore.",
        existing_source="I live in Bangalore.",
        incoming_fact="The user lives in Chennai.",
        incoming_source="I moved to Chennai last week.",
        existing_timestamp=stamp,
    )
    assert decision.existing_timestamp == stamp