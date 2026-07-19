"""
Gate-level test set.

Measures what the product actually outputs — gate verdicts — rather than raw
scorer labels. The distinction matters: the attribution fix lives in the gate,
so a scorer-level test cannot see it. The NLI model still returns 0.99
"faithful" for "my brother is a lawyer"; the gate is what overrides it.

Expected verdicts:
    STORE   the fact is supported and should enter memory
    REVIEW  uncertain or possibly misattributed — a human decides
    REJECT  the fact contradicts its source

Categories:
    entailed     directly supported by the source
    implied      supported after one inferential step
    unsupported  not derivable from the source, but not contradictory
    contradicts  conflicts with the source
    attribution  the fact belongs to a third party, not the speaker
"""

STORE = "STORE"
REVIEW = "REVIEW"
REJECT = "REJECT"

# (source, fact, expected_verdict, category, note)
GATE_CASES = [

    # ---- entailed → STORE ---------------------------------------------
    (
        "I am allergic to peanuts.",
        "The user is allergic to peanuts.",
        STORE,
        "entailed",
        "Near-verbatim restatement.",
    ),
    (
        "I work as a nurse at the city hospital.",
        "The user is a nurse.",
        STORE,
        "entailed",
        "Simple extraction of a stated fact.",
    ),
    (
        "I have two dogs named Max and Bella.",
        "The user has two dogs.",
        STORE,
        "entailed",
        "Quantity and subject stated explicitly.",
    ),
    (
        "I don't drink alcohol.",
        "The user does not drink alcohol.",
        STORE,
        "entailed",
        "Negation preserved.",
    ),

    # ---- implied → STORE ----------------------------------------------
    (
        "I moved from Delhi to Bangalore last month.",
        "The user lives in Bangalore.",
        STORE,
        "implied",
        "KNOWN FAILURE: model returns contradiction 0.77. Gate rejects a true "
        "fact — silent data loss. Sub-task 2.",
    ),
    (
        "I just finished medical school and started practising.",
        "The user is a doctor.",
        STORE,
        "implied",
        "One inferential step. Currently passes.",
    ),
    (
        "I've been vegetarian since I was twelve.",
        "The user does not eat meat.",
        STORE,
        "implied",
        "Definitional inference. Currently passes.",
    ),
    (
        "My commute takes 45 minutes each way.",
        "The user does not work from home.",
        STORE,
        "implied",
        "Negative inference. Currently passes.",
    ),

    # ---- unsupported → REVIEW -----------------------------------------
    (
        "I had a great salad for lunch.",
        "The user is a doctor.",
        REVIEW,
        "unsupported",
        "KNOWN FAILURE: model returns contradiction 0.99 on unrelated content. "
        "Sub-task 2.",
    ),
    (
        "I live in an apartment on the fifth floor.",
        "The user owns a car.",
        REVIEW,
        "unsupported",
        "KNOWN FAILURE: contradiction over-fires on unrelated pairs. Sub-task 2.",
    ),
    (
        "I enjoy hiking on weekends.",
        "The user has three children.",
        REVIEW,
        "unsupported",
        "Correctly neutral. Currently passes.",
    ),

    # ---- contradicts → REJECT -----------------------------------------
    (
        "I moved from Delhi to Bangalore last month.",
        "The user lives in Delhi.",
        REJECT,
        "contradicts",
        "Direct conflict with the stated move.",
    ),
    (
        "I am allergic to peanuts.",
        "The user loves eating peanuts.",
        REJECT,
        "contradicts",
        "Life-critical contradiction. The headline case.",
    ),
    (
        "I'm vegetarian.",
        "The user eats meat.",
        REJECT,
        "contradicts",
        "Direct negation.",
    ),
    (
        "I graduated from MIT with a degree in physics.",
        "The user never went to college.",
        REJECT,
        "contradicts",
        "Contradicts an explicit claim.",
    ),

    # ---- attribution → REVIEW -----------------------------------------
    (
        "I was talking about my neighbour's cat.",
        "The user has a cat.",
        REVIEW,
        "attribution",
        "Scorer says 0.98 faithful. Gate must catch the misattribution.",
    ),
    (
        "My brother is a lawyer.",
        "The user is a lawyer.",
        REVIEW,
        "attribution",
        "Scorer says 0.99 faithful. Profession belongs to a third party.",
    ),
    (
        "My colleague mentioned she is moving to Berlin.",
        "The user is moving to Berlin.",
        REVIEW,
        "attribution",
        "Scorer says 0.99 faithful. Reported speech about a third party.",
    ),
]