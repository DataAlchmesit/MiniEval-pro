"""
Diagnostic: does semantic similarity predict spurious contradictions?

Hypothesis
----------
The NLI model returns high "contradiction" on pairs that are merely unrelated,
because its three labels leave no room for "these texts have nothing to do with
each other." If that is what is happening, unrelated pairs should score LOW on
semantic similarity, while genuine contradictions should score HIGH — the two
statements are about the same thing, they just disagree.

If the hypothesis holds, a relatedness pre-check can downgrade spurious
contradictions to neutral.

If genuine and spurious contradictions have overlapping similarity ranges,
the approach cannot separate them and should be abandoned.

    python -m minieval_pro.tests.diagnose_contradiction
"""

from __future__ import annotations

from ..scorers.faithfulness import FaithfulnessScorer
from ..scorers.relevance import RelevanceScorer


# (source, fact, kind, expectation)
PAIRS = [
    # --- spurious: model says contradiction, but nothing contradicts -----
    (
        "I had a great salad for lunch.",
        "The user is a doctor.",
        "spurious",
        "Unrelated. Should score LOW similarity if hypothesis holds.",
    ),
    (
        "I live in an apartment on the fifth floor.",
        "The user owns a car.",
        "spurious",
        "Unrelated. Should score LOW similarity if hypothesis holds.",
    ),
    (
        "I moved from Delhi to Bangalore last month.",
        "The user lives in Bangalore.",
        "spurious",
        "Related but misread. Expected to score HIGH — the hard case.",
    ),

    # --- genuine: real contradictions, must keep being rejected ----------
    (
        "I moved from Delhi to Bangalore last month.",
        "The user lives in Delhi.",
        "genuine",
        "Real contradiction. Should score HIGH similarity.",
    ),
    (
        "I am allergic to peanuts.",
        "The user loves eating peanuts.",
        "genuine",
        "Real contradiction. Should score HIGH similarity.",
    ),
    (
        "I'm vegetarian.",
        "The user eats meat.",
        "genuine",
        "Real contradiction. Should score HIGH similarity.",
    ),
    (
        "I graduated from MIT with a degree in physics.",
        "The user never went to college.",
        "genuine",
        "Real contradiction. Should score HIGH similarity.",
    ),

    # --- control: correctly neutral, for comparison ----------------------
    (
        "I enjoy hiking on weekends.",
        "The user has three children.",
        "control-neutral",
        "Model already says neutral. Unrelated pair.",
    ),

    # --- control: correctly faithful, must not be disturbed --------------
    (
        "I am allergic to peanuts.",
        "The user is allergic to peanuts.",
        "control-faithful",
        "Model already correct. Should score HIGH similarity.",
    ),
]


def run() -> None:
    faith = FaithfulnessScorer()
    rel = RelevanceScorer()

    rows = []
    for source, fact, kind, note in PAIRS:
        f = faith.score(context=source, answer=fact)
        r = rel.score(source, fact)
        rows.append(
            {
                "source": source,
                "fact": fact,
                "kind": kind,
                "label": f.label,
                "contradiction": getattr(f, "contradiction", None),
                "similarity": r.score,
                "note": note,
            }
        )

    print()
    print("=" * 84)
    print("  DIAGNOSTIC — does similarity separate spurious from genuine contradictions?")
    print("=" * 84)

    current = None
    for row in rows:
        if row["kind"] != current:
            current = row["kind"]
            print(f"\n--- {current.upper()} ---")
        contra = row["contradiction"]
        contra_str = f"{contra:.3f}" if contra is not None else "  n/a"
        print(f"\n  similarity={row['similarity']:.3f}   contradiction={contra_str}   label={row['label']}")
        print(f"      source: {row['source']}")
        print(f"      fact:   {row['fact']}")

    # Compare the ranges — this is the actual question.
    spurious = [r["similarity"] for r in rows if r["kind"] == "spurious"]
    genuine = [r["similarity"] for r in rows if r["kind"] == "genuine"]

    print()
    print("=" * 84)
    print("  VERDICT")
    print("=" * 84)
    print(f"  spurious similarity range: {min(spurious):.3f} - {max(spurious):.3f}")
    print(f"  genuine  similarity range: {min(genuine):.3f} - {max(genuine):.3f}")
    print()

    if max(spurious) < min(genuine):
        threshold = (max(spurious) + min(genuine)) / 2
        print(f"  SEPARABLE. A threshold near {threshold:.3f} splits them cleanly.")
        print("  The relatedness pre-check should work.")
    else:
        overlap_lo = max(min(spurious), min(genuine))
        overlap_hi = min(max(spurious), max(genuine))
        print(f"  OVERLAP between {overlap_lo:.3f} and {overlap_hi:.3f}.")
        print("  A single similarity threshold cannot separate these cleanly.")
        print("  Check whether excluding the Bangalore case changes this — it may be")
        print("  a different failure mode that needs its own treatment.")

        # Re-check without the known-hard related case.
        spurious_unrelated = [
            r["similarity"] for r in rows
            if r["kind"] == "spurious" and "Bangalore last month" not in r["source"]
        ]
        if spurious_unrelated and max(spurious_unrelated) < min(genuine):
            t = (max(spurious_unrelated) + min(genuine)) / 2
            print()
            print(f"  Excluding the related-but-misread case: SEPARABLE near {t:.3f}.")
            print("  Relatedness fixes the unrelated pairs; Bangalore needs a different fix.")

    print()


if __name__ == "__main__":
    run()