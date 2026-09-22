"""
Tests AttributionHook — confirms the tri-state contract
(accepted/rejected/uncertain) correctly reflects the underlying
MemoryGate.check() verdicts (STORE/REJECT/REVIEW), which are already
tested and verified elsewhere (test_regression.py, test_attribution_gaps.py,
test_attribution_narrowing.py).

This file tests the TRANSLATION LAYER specifically — does the hook map
each real verdict to the right tri-state label, with a non-empty reason,
and does metadata pass through without breaking anything. It does not
re-test the underlying attribution/faithfulness logic itself.

Run with:
    pytest minieval_pro/tests/test_attribution_hook.py -v
"""

from dataclasses import dataclass

import pytest

from minieval_pro.gate import MemoryGate
from minieval_pro.hooks.attribution_hook import AttributionHook, ACCEPTED, REJECTED, UNCERTAIN


@dataclass
class HookCase:
    label: str
    candidate: str
    source_text: str
    expected_verdict: str
    note: str = ""


CASES = [
    HookCase(
        label="clean_store_maps_to_accepted",
        candidate="The user is allergic to peanuts.",
        source_text="I am allergic to peanuts.",
        expected_verdict=ACCEPTED,
        note="Direct, faithful fact — confirmed STORE via check() earlier "
             "this session (0.99 faithful).",
    ),
    HookCase(
        label="clean_reject_maps_to_rejected",
        candidate="The user loves eating peanuts.",
        source_text="I am allergic to peanuts.",
        expected_verdict=REJECTED,
        note="Direct contradiction — confirmed REJECT via check() earlier "
             "this session (0.00 contradicts).",
    ),
    HookCase(
        label="her_brother_maps_to_uncertain",
        candidate="The user is a lawyer.",
        source_text="Her brother is a lawyer and he helped with the contract.",
        expected_verdict=UNCERTAIN,
        note="The exact case from the attribution guard fix earlier "
             "today — confirmed REVIEW via check() after the possessive "
             "pronoun fix. This is the primary case the hook exists for.",
    ),
    HookCase(
        label="their_neighbour_maps_to_uncertain",
        candidate="The user owns a cat.",
        source_text="Their neighbour's cat keeps getting into the garden.",
        expected_verdict=UNCERTAIN,
        note="Confirmed REVIEW via check() after the possessive fix.",
    ),
    HookCase(
        label="plain_third_person_maps_to_uncertain",
        candidate="The user is a lawyer.",
        source_text="She works as a lawyer downtown.",
        expected_verdict=UNCERTAIN,
        note="No possessive, no reporting verb — confirmed REVIEW via "
             "check() after the third-person-subject fix.",
    ),
]


@pytest.fixture(scope="module")
def hook():
    return AttributionHook(gate=MemoryGate(quiet=True))


@pytest.mark.parametrize("case", CASES, ids=[c.label for c in CASES])
def test_hook_verdict_matches_expected(hook, case):
    result = hook.check(candidate=case.candidate, source_text=case.source_text)

    print(f"\n  verdict={result.verdict}  reason={result.reason!r}")

    assert result.verdict == case.expected_verdict, (
        f"{case.label}: expected {case.expected_verdict}, got "
        f"{result.verdict}. {case.note}"
    )

    # Every verdict must carry a real reason — an "uncertain" with no
    # explanation isn't actionable for whoever routes it downstream.
    assert result.reason, f"{case.label}: reason should not be empty"

    # raw_decision should always be the real underlying GateDecision,
    # not a stub — callers relying on it for audit need the real object.
    assert result.raw_decision is not None
    assert result.raw_decision.verdict in ("STORE", "REJECT", "REVIEW")


def test_metadata_accepted_but_does_not_change_verdict(hook):
    """
    metadata is currently unused by the guard logic (reserved for future
    role/actor_id integration — see attribution_hook.py docstring). This
    test confirms passing it doesn't break anything or silently change
    behavior, which would be a surprising, undocumented side effect.
    """
    candidate = "The user is allergic to peanuts."
    source_text = "I am allergic to peanuts."

    without_metadata = hook.check(candidate=candidate, source_text=source_text)
    with_metadata = hook.check(
        candidate=candidate,
        source_text=source_text,
        metadata={"user_id": "alice", "role": "user"},
    )

    assert without_metadata.verdict == with_metadata.verdict
    assert without_metadata.verdict == ACCEPTED


def test_every_case_resolves_to_one_of_three_states(hook):
    """
    Never a silent pass — every candidate must resolve to exactly one of
    accepted/rejected/uncertain, per the contract in mem0ai/mem0#7283.
    """
    valid_verdicts = {ACCEPTED, REJECTED, UNCERTAIN}
    for case in CASES:
        result = hook.check(candidate=case.candidate, source_text=case.source_text)
        assert result.verdict in valid_verdicts, (
            f"{case.label} produced an invalid verdict: {result.verdict!r}"
        )