"""
Seeds a FRESH audit log with real, multi-sentence cases so the dashboard
can be tested against data that actually exercises evidence_span.

Most of the existing regression corpus (attribution_corpus.py) uses
single-sentence sources ("My brother is a lawyer and he helped me with
the contract." is one sentence by split_sentences' period-based
definition, despite being a compound clause) — evidence_span is always
None on those, by design, since there's nothing to narrow. This script
uses genuinely multi-sentence sources instead, the shape that actually
exercises the narrowing guards.

Writes to a NEW file (dashboard_demo_fresh.jsonl), not demo_audit.jsonl,
so the existing legacy data is untouched and this log is fully hashed
from its first entry — a different, useful case to see the chain badge
render ("N hashed, 0 legacy") versus what's already been seen
("0 hashed, N legacy").

Run with:
    python seed_dashboard_demo.py

Then point the dashboard at it:
    set MINIEVAL_AUDIT_PATH=dashboard_demo_fresh.jsonl   (Windows)
    python dashboard\\app.py
"""

from minieval_pro.gate import MemoryGate
from minieval_pro.persistence import AuditLog

CASES = [
    # Real case from test_attribution_narrowing.py -- three real sentences,
    # one source, three facts extracted from it. This is the exact shape
    # that proves evidence_span isolates the right sentence per fact.
    {
        "source": (
            "I moved to Chennai last week. I had a great salad for lunch. "
            "My brother is a lawyer and he helped me with the contract."
        ),
        "fact": "The user lives in Chennai.",
    },
    {
        "source": (
            "I moved to Chennai last week. I had a great salad for lunch. "
            "My brother is a lawyer and he helped me with the contract."
        ),
        "fact": "The user loves eating peanuts.",
    },
    {
        "source": (
            "I moved to Chennai last week. I had a great salad for lunch. "
            "My brother is a lawyer and he helped me with the contract."
        ),
        "fact": "The user is a lawyer.",
    },
    # A second, different multi-sentence source, for variety in the chart
    # and the table -- not the same three sentences recycled.
    {
        "source": (
            "I used to live in Delhi. I switched jobs last year. "
            "My sister teaches high school and loves it."
        ),
        "fact": "The user teaches high school.",
    },
    {
        "source": (
            "I used to live in Delhi. I switched jobs last year. "
            "My sister teaches high school and loves it."
        ),
        "fact": "The user switched jobs last year.",
    },
    # One clean, direct single-sentence STORE -- confirms evidence_span is
    # correctly None here (nothing to narrow), by contrast with the
    # multi-sentence cases above.
    {
        "source": "I am allergic to peanuts.",
        "fact": "The user is allergic to peanuts.",
    },
]


def run():
    gate = MemoryGate(quiet=True)
    log = AuditLog("dashboard_demo_fresh.jsonl")

    print("Seeding dashboard_demo_fresh.jsonl with real, multi-sentence cases...\n")

    for case in CASES:
        decision = gate.check(source=case["source"], fact=case["fact"])
        log.record(decision)
        span_note = (
            f"  evidence_span: \"{decision.evidence_span}\""
            if decision.evidence_span
            else "  evidence_span: None (single-sentence source, nothing to narrow)"
        )
        print(f"[{decision.verdict}] \"{case['fact']}\"")
        print(span_note)
        print()

    ok, report = log.verify_chain()
    print(f"Chain check: ok={ok}, report={report}")


if __name__ == "__main__":
    run()