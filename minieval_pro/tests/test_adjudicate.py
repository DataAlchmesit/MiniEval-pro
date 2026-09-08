"""
Regression suite for gate.adjudicate().

Field names below match the real AdjudicationDecision in gate.py:
verdict is "ACCEPT" (ACCEPT_OVERWRITE), "BLOCK" (BLOCK_OVERWRITE), or "REVIEW".

Run with:
    python test_adjudicate.py
"""

from dataclasses import dataclass
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

    # --- Case 1: same fact, said in different words ---
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

    # --- Case 2: genuine update, both facts faithful to their own source ---
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

    # --- Case 3: old fact true, incoming fact hallucinated ---
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
            "right outcome, but for an unverified reason. Worth writing "
            "up as a known limitation, same spirit as the check() "
            "Delhi/Bangalore case."
        ),
    ),

    # --- Case 4: both facts weakly supported — should never silently pick a winner ---
    AdjudicateCase(
        label="case_4_both_weak",
        existing_fact="The user probably prefers tea over coffee.",
        existing_source="Yeah I guess I might lean towards tea sometimes.",
        incoming_fact="The user prefers coffee over tea.",
        incoming_source="I don't know, coffee's fine I suppose.",
        expected="REVIEW",
        note=(
            "Neither source clearly supports its own fact — both are "
            "hedgy, low-confidence statements. adjudicate() only checks "
            "whether the INCOMING fact clears the faithfulness bar; it "
            "never compares existing_faithfulness against incoming_faithfulness. "
            "So if incoming happens to score just above store_threshold, "
            "it could ACCEPT and overwrite an existing memory of similar "
            "or even higher confidence. This case is designed to surface "
            "that — if it fails, it's not a test bug, it's the real gap "
            "found by reading gate.py directly: existing_faithfulness is "
            "computed and stored on the decision but never used in the "
            "verdict logic."
        ),
    ),

    # --- Case 4b: incoming weakly faithful, existing strongly faithful ---
    # This is the sharper attempt at the same gap as case 4. Case 4 fell
    # through to REVIEW because both sides scored 0.00/neutral — it never
    # actually tested the threshold boundary. This case tries to land
    # incoming just above store_threshold while existing is much stronger,
    # to see if adjudicate() ACCEPTs an overwrite it probably shouldn't.
    #
    # IMPORTANT: exact scores depend on the real NLI model and can't be
    # guaranteed by hand. Run this, read the actual existing_faithfulness
    # and incoming_faithfulness printed below, and treat "expected" as a
    # hypothesis to check against the real numbers — not a certainty.
    AdjudicateCase(
        label="case_4b_asymmetric_evidence",
        existing_fact="The user lives in France.",
        existing_source="I live in France.",
        incoming_fact="The user does not live in France.",
        incoming_source="I think I might live in France one day.",
        expected="REVIEW",
        note=(
            "Third attempt at the asymmetric-evidence gap. Cases 4 and "
            "4a/4b(v1) both fell through to neutral/neutral because the "
            "source-fact pairs were inferential or hedged in the SOURCE. "
            "This pair is direct (existing) and hedged in the FACT itself "
            "(incoming), matching the pattern that scored 0.99 in case 3's "
            "existing pair. Predicted: existing lands high faithful "
            "(~0.95+, near-identical wording to case 1/2's Bangalore "
            "pair), incoming lands mid-range (~0.5-0.65, plausible but "
            "clearly hedged). If incoming ACCEPTs, the gap is confirmed: "
            "adjudicate() never compares existing_faithfulness to "
            "incoming_faithfulness. If incoming again lands as neutral, "
            "read the real printed scores before concluding anything —  "
            "do not edit gate.py based on a hypothesis that hasn't been "
            "confirmed by an actual run."
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
            "Fourth and final attempt at the asymmetric-evidence gap "
            "before treating it as documented-but-unconfirmed. Existing "
            "is direct and declarative (should land high faithful, "
            "similar to the France/Bangalore pairs). Incoming is a plain "
            "declarative fact, not hedged with 'might' or 'I think' "
            "(unlike 4b, which hedged and scored 0.36) — but the source "
            "only weakly implies the fact (painting a room green doesn't "
            "establish it as a favorite color), so it may land faithful "
            "but moderate. If incoming clears store_threshold (0.5) and "
            "ACCEPTs, the gap is finally confirmed empirically. If this "
            "also fails to land in the zone, stop testing by hand and "
            "document the gap as theoretical-but-unconfirmed in "
            "CHANGELOG.md / README.md, same tone as the Delhi/Bangalore "
            "and adjudicate() relatedness-guard limitations."
        ),
    ),

    # --- Case 5: the EXISTING fact was itself weakly supported when written ---
    AdjudicateCase(
        label="case_5_existing_was_weak",
        existing_fact="The user works as a teacher.",
        existing_source="I think I mentioned I do some tutoring sometimes.",
        incoming_fact="The user works as a software engineer.",
        incoming_source="I've been a software engineer for six years now.",
        expected="ACCEPT",
        note=(
            "Existing was only ever weakly supported (hedged, vague "
            "source). Incoming is direct and well-supported. Should "
            "ACCEPT cleanly under current logic, since adjudicate() only "
            "checks incoming's own faithfulness and doesn't examine how "
            "strong existing was. Worth confirming existing_faithfulness "
            "actually printed low, to know this tested what it was "
            "meant to."
        ),
    ),

    # --- Case 6: partial / more specific update, not a real contradiction ---
    AdjudicateCase(
        label="case_6_more_specific_not_contradictory",
        existing_fact="The user lives in Bangalore.",
        existing_source="I live in Bangalore.",
        incoming_fact="The user lives in Koramangala, Bangalore.",
        incoming_source="I live in Koramangala, it's a neighborhood in Bangalore.",
        expected="ACCEPT",
        note=(
            "Not a real conflict — incoming is more specific, not "
            "contradictory. Both should score faithful against their own "
            "source, so this should ACCEPT the more detailed version "
            "cleanly. If the NLI model reads the added specificity as "
            "unrelated/neutral instead, this might land as REVIEW — a "
            "real, if minor, finding: adjudicate() has no notion of "
            "'this is a refinement, not a replacement.'"
        ),
    ),

]

def test_existing_timestamp_passthrough():
    """
    Confirms existing_timestamp is optional and passes through correctly.
    """
    gate = MemoryGate(quiet=True)

    # 1. Omitted — should default to None.
    decision_without = gate.adjudicate(
        existing_fact="The user lives in Bangalore.",
        existing_source="I live in Bangalore.",
        incoming_fact="The user lives in Chennai.",
        incoming_source="I moved to Chennai last week.",
    )
    without_ok = decision_without.existing_timestamp is None

    # 2. Supplied — should come back unchanged.
    stamp = "2026-03-04T10:15:00+00:00"
    decision_with = gate.adjudicate(
        existing_fact="The user lives in Bangalore.",
        existing_source="I live in Bangalore.",
        incoming_fact="The user lives in Chennai.",
        incoming_source="I moved to Chennai last week.",
        existing_timestamp=stamp,
    )
    with_ok = decision_with.existing_timestamp == stamp

    print("\n" + "=" * 70)
    print("existing_timestamp passthrough check")
    print("=" * 70)
    print(f"  [{'PASS' if without_ok else 'FAIL'}] omitted -> None "
          f"(got: {decision_without.existing_timestamp!r})")
    print(f"  [{'PASS' if with_ok else 'FAIL'}] supplied -> unchanged "
          f"(got: {decision_with.existing_timestamp!r})")

    # Replace 'return without_ok and with_ok' with explicit asserts:
    assert without_ok, "existing_timestamp was expected to be None when omitted"
    assert with_ok, f"existing_timestamp was expected to be {stamp!r}"

def run_suite():
    gate = MemoryGate(quiet=True)
    results = []

    for case in CASES:
        decision = gate.adjudicate(
            existing_fact=case.existing_fact,
            existing_source=case.existing_source,
            incoming_fact=case.incoming_fact,
            incoming_source=case.incoming_source,
        )
        actual = decision.verdict
        passed = (actual == case.expected)
        results.append((case, actual, passed, decision))

    return results


def print_report(results):
    total = len(results)
    passed = sum(1 for _, _, ok, _ in results if ok)

    print("=" * 70)
    print("MiniEval adjudicate() regression suite")
    print("=" * 70)

    for case, actual, ok, decision in results:
        mark = "PASS" if ok else "FAIL"
        print(f"\n[{mark}] {case.label}")
        print(f"  existing: \"{case.existing_fact}\"  <- \"{case.existing_source}\"")
        print(f"  incoming: \"{case.incoming_fact}\"  <- \"{case.incoming_source}\"")
        print(f"  expected={case.expected}  actual={actual}")
        print(f"  existing_faithfulness={decision.existing_faithfulness:.2f} "
              f"({decision.existing_label})   "
              f"incoming_faithfulness={decision.incoming_faithfulness:.2f} "
              f"({decision.incoming_label})")
        if not ok and case.note:
            print(f"  note: {case.note}")

    print("\n" + "-" * 70)
    print(f"Total: {total}   Passed: {passed}   Failed: {total - passed}")
    print("-" * 70)


if __name__ == "__main__":
    results = run_suite()
    print_report(results)
    test_existing_timestamp_passthrough()