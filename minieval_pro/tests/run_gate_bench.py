"""
Run the gate-level test set and report accuracy by category.

This measures what the product outputs — STORE / REVIEW / REJECT — rather than
raw scorer labels, so fixes that live in the gate (such as the attribution
pre-check) are visible.

    python -m minieval_pro.tests.run_gate_bench
"""

from __future__ import annotations

from collections import defaultdict

from ..gate import MemoryGate
from .gate_cases import GATE_CASES


def run() -> dict:
    gate = MemoryGate()

    results = []
    by_category: dict[str, list[bool]] = defaultdict(list)

    for source, fact, expected, category, note in GATE_CASES:
        decision = gate.check(source=source, fact=fact)
        correct = decision.verdict == expected
        by_category[category].append(correct)
        results.append(
            {
                "source": source,
                "fact": fact,
                "expected": expected,
                "actual": decision.verdict,
                "faithfulness": decision.faithfulness,
                "label": decision.label,
                "reason": decision.reason,
                "category": category,
                "correct": correct,
                "note": note,
            }
        )

    total = len(results)
    passed = sum(1 for r in results if r["correct"])

    return {
        "results": results,
        "total": total,
        "passed": passed,
        "accuracy": round(100 * passed / total, 1) if total else 0.0,
        "by_category": {
            cat: {
                "passed": sum(vals),
                "total": len(vals),
                "accuracy": round(100 * sum(vals) / len(vals), 1),
            }
            for cat, vals in by_category.items()
        },
    }


def report(data: dict) -> None:
    print()
    print("=" * 78)
    print("  GATE TEST SET  —  measuring verdicts, not scorer labels")
    print("=" * 78)

    current = None
    for r in data["results"]:
        if r["category"] != current:
            current = r["category"]
            print(f"\n--- {current.upper()} ---")

        mark = "PASS" if r["correct"] else "FAIL"
        print(f"\n  [{mark}] expected={r['expected']:<7} actual={r['actual']:<7} "
              f"faith={r['faithfulness']:.2f} ({r['label']})")
        print(f"        source: {r['source']}")
        print(f"        fact:   {r['fact']}")
        print(f"        reason: {r['reason']}")
        if not r["correct"]:
            print(f"        note:   {r['note']}")

    print()
    print("=" * 78)
    print("  BY CATEGORY")
    print("=" * 78)
    for cat, stats in data["by_category"].items():
        print(f"  {cat:<14} {stats['passed']:>2}/{stats['total']:<2}  {stats['accuracy']:>5.1f}%")

    print()
    print("-" * 78)
    print(f"  OVERALL       {data['passed']:>2}/{data['total']:<2}  {data['accuracy']:>5.1f}%")
    print("-" * 78)
    print()


if __name__ == "__main__":
    report(run())