"""
Run the faithfulness test set and report accuracy by category.

Use this before and after any scoring change to show the effect rather than
assume it.

    python -m minieval_pro.tests.run_faithfulness_bench
"""

from __future__ import annotations

from collections import defaultdict

from ..scorers.faithfulness import FaithfulnessScorer
from .test_cases import TEST_CASES


def run() -> dict:
    scorer = FaithfulnessScorer()

    results = []
    by_category: dict[str, list[bool]] = defaultdict(list)

    for source, fact, expected_label, category, note in TEST_CASES:
        r = scorer.score(context=source, answer=fact)
        correct = r.label == expected_label
        by_category[category].append(correct)
        results.append(
            {
                "source": source,
                "fact": fact,
                "expected": expected_label,
                "actual": r.label,
                "score": r.score,
                "entailment": getattr(r, "entailment", None),
                "contradiction": getattr(r, "contradiction", None),
                "neutral": getattr(r, "neutral", None),
                "category": category,
                "correct": correct,
                "note": note,
            }
        )

    total = len(results)
    passed = sum(1 for x in results if x["correct"])

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
    print("  FAITHFULNESS TEST SET")
    print("=" * 78)

    current_cat = None
    for r in data["results"]:
        if r["category"] != current_cat:
            current_cat = r["category"]
            print(f"\n--- {current_cat.upper()} ---")

        mark = "PASS" if r["correct"] else "FAIL"
        print(f"\n  [{mark}] expected={r['expected']:<12} actual={r['actual']:<12} score={r['score']:.2f}")
        print(f"        source: {r['source']}")
        print(f"        fact:   {r['fact']}")
        if r["entailment"] is not None:
            print(
                f"        raw:    entail={r['entailment']:.3f} "
                f"contra={r['contradiction']:.3f} neutral={r['neutral']:.3f}"
            )
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