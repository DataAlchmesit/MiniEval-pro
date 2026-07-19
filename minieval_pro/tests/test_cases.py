"""
Faithfulness test set for M3.

Purpose: measure scoring behaviour before and after changes, so improvements
are demonstrated rather than assumed.

Each case carries the expected LABEL, not an expected score. Labels are the
stable contract; exact scores are implementation detail and will move as the
formula changes.

Categories:
    entailed    the fact is directly supported by the source
    implied     a human would call it supported, but it is not restated verbatim
                (this is where the model currently fails)
    unsupported the fact is not derivable from the source, but does not
                contradict it either — genuinely neutral
    contradicts the fact conflicts with the source
    attribution the fact misattributes something in the source to the user
                (the "neighbour's cat" family of errors)
"""

# (source, fact, expected_label, category, note)
TEST_CASES = [

    # ---- entailed: directly supported, should already pass -------------
    (
        "I am allergic to peanuts.",
        "The user is allergic to peanuts.",
        "faithful",
        "entailed",
        "Near-verbatim restatement. Baseline sanity check.",
    ),
    (
        "I work as a nurse at the city hospital.",
        "The user is a nurse.",
        "faithful",
        "entailed",
        "Simple extraction of a stated fact.",
    ),
    (
        "I have two dogs named Max and Bella.",
        "The user has two dogs.",
        "faithful",
        "entailed",
        "Quantity and subject both stated explicitly.",
    ),
    (
        "I don't drink alcohol.",
        "The user does not drink alcohol.",
        "faithful",
        "entailed",
        "Negation preserved correctly.",
    ),

    # ---- implied: the inference gap, currently scored neutral ----------
    (
        "I moved from Delhi to Bangalore last month.",
        "The user lives in Bangalore.",
        "faithful",
        "implied",
        "Moving to X implies living in X. Currently scored neutral at high confidence.",
    ),
    (
        "I just finished medical school and started practising.",
        "The user is a doctor.",
        "faithful",
        "implied",
        "Requires one inferential step from the stated fact.",
    ),
    (
        "I've been vegetarian since I was twelve.",
        "The user does not eat meat.",
        "faithful",
        "implied",
        "Definitional inference — vegetarian entails not eating meat.",
    ),
    (
        "My commute takes 45 minutes each way.",
        "The user does not work from home.",
        "faithful",
        "implied",
        "Negative inference. Harder: requires ruling something out.",
    ),

    # ---- unsupported: genuinely neutral, should NOT be rejected --------
    (
        "I had a great salad for lunch.",
        "The user is a doctor.",
        "neutral",
        "unsupported",
        "Unrelated to source. Not a contradiction — just unsupported.",
    ),
    (
        "I live in an apartment on the fifth floor.",
        "The user owns a car.",
        "neutral",
        "unsupported",
        "Plausible but unstated. Must not be treated as a contradiction.",
    ),
    (
        "I enjoy hiking on weekends.",
        "The user has three children.",
        "neutral",
        "unsupported",
        "Completely orthogonal fact.",
    ),

    # ---- contradicts: must be caught ----------------------------------
    (
        "I moved from Delhi to Bangalore last month.",
        "The user lives in Delhi.",
        "contradicts",
        "contradicts",
        "Direct conflict with the stated move.",
    ),
    (
        "I am allergic to peanuts.",
        "The user loves eating peanuts.",
        "contradicts",
        "contradicts",
        "Life-critical contradiction. The headline case.",
    ),
    (
        "I'm vegetarian.",
        "The user eats meat.",
        "contradicts",
        "contradicts",
        "Direct negation of a stated fact.",
    ),
    (
        "I graduated from MIT with a degree in physics.",
        "The user never went to college.",
        "contradicts",
        "contradicts",
        "Contradicts an explicit claim.",
    ),

    # ---- attribution: the possessive / entity-resolution failure ------
    (
        "I was talking about my neighbour's cat.",
        "The user has a cat.",
        "neutral",
        "attribution",
        "Scored 0.96 faithful previously. Model keyword-matches 'cat' and "
        "misses that the cat belongs to someone else.",
    ),
    (
        "My brother is a lawyer.",
        "The user is a lawyer.",
        "neutral",
        "attribution",
        "Profession belongs to a third party, not the speaker.",
    ),
    (
        "My colleague mentioned she is moving to Berlin.",
        "The user is moving to Berlin.",
        "neutral",
        "attribution",
        "Reported speech about a third party.",
    ),
]


def summary() -> dict:
    """Count cases by expected label and by category."""
    by_label: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for _, _, label, category, _ in TEST_CASES:
        by_label[label] = by_label.get(label, 0) + 1
        by_category[category] = by_category.get(category, 0) + 1
    return {
        "total": len(TEST_CASES),
        "by_label": by_label,
        "by_category": by_category,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))