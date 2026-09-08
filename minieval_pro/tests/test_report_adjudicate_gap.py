"""
Proves how generate_report() handles AdjudicationDecision entries, not just
GateDecision ones. Uses plain dicts shaped exactly like the real
to_dict() output of each, so no models need to load — this only tests
the report renderer itself.

Run with:
    python test_report_adjudicate_gap.py
"""

import tempfile
from pathlib import Path

from minieval_pro.persistence import AuditLog, generate_report


def make_log():
    """Build a small audit log with both decision types mixed together."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    )
    tmp.close()
    log = AuditLog(tmp.name)

    # A normal check() REVIEW — should categorise as "unrelated" via the
    # existing string-match logic.
    log.record({
        "verdict": "REVIEW",
        "fact": "The user is a doctor.",
        "source": "I had a salad for lunch today.",
        "reason": "Source and fact appear unrelated (similarity 0.30 < 0.58).",
        "policy_name": "default",
        "policy_version": "1.0",
        "policy_fingerprint": "abc123",
        "timestamp": "2026-09-01T10:00:00+00:00",
    })

    # A normal check() REJECT — should show up in "blocked" with fact/source.
    log.record({
        "verdict": "REJECT",
        "fact": "The user loves eating peanuts.",
        "source": "I am allergic to peanuts.",
        "reason": "Fact contradicts its source.",
        "policy_name": "default",
        "policy_version": "1.0",
        "policy_fingerprint": "abc123",
        "timestamp": "2026-09-01T10:05:00+00:00",
        "contradiction": 0.99,
    })

    # An adjudicate() REVIEW — this is the case in question. Real
    # AdjudicationDecision.to_dict() output has existing_fact/incoming_fact,
    # NOT fact/source.
    log.record({
        "verdict": "REVIEW",
        "existing_fact": "The user prefers tea.",
        "existing_faithfulness": 0.0,
        "existing_label": "neutral",
        "incoming_fact": "The user prefers coffee.",
        "incoming_faithfulness": 0.0,
        "incoming_label": "neutral",
        "reason": "Incoming memory is not clearly faithful. Not confident "
                  "enough to overwrite an existing memory.",
        "policy_name": "default",
        "policy_version": "1.0",
        "policy_fingerprint": "abc123",
        "timestamp": "2026-09-01T10:10:00+00:00",
    })

    # An adjudicate() BLOCK — real verdict string is "BLOCK", not "REJECT".
    log.record({
        "verdict": "BLOCK",
        "existing_fact": "The user is allergic to peanuts.",
        "incoming_source": "I had a salad for lunch.",
        "existing_faithfulness": 0.99,
        "existing_label": "faithful",
        "incoming_fact": "The user loves eating peanuts.",
        "existing_source": "I am allergic to peanuts.",
        "incoming_faithfulness": 0.0,
        "incoming_label": "contradicts",
        "reason": "Incoming memory contradicts its own source. Existing "
                  "memory protected.",
        "policy_name": "default",
        "policy_version": "1.0",
        "policy_fingerprint": "abc123",
        "timestamp": "2026-09-01T10:15:00+00:00",
    })

    return log


def run():
    log = make_log()
    report = generate_report(log)
    print(report)

    print("\n" + "=" * 70)
    print("What to check by eye:")
    print("=" * 70)
    print("1. Does 'Blocked' show 2 entries (the REJECT and the BLOCK), "
          "or only 1 (just the REJECT)?")
    print("2. For the adjudicate() BLOCK entry, does the fact/source text "
          "show real content, or blank/empty?")
    print("3. Under 'Why facts were flagged', does the adjudicate() REVIEW "
          "get its own accurate category, or does it fall into "
          "'Neither supported nor contradicted' (a check()-shaped label "
          "being applied to an adjudicate() reason)?")

    Path(log.path).unlink(missing_ok=True)


if __name__ == "__main__":
    run()