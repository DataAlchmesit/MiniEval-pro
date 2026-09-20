"""
Attribution checking — who does this fact actually belong to?

The NLI model has no notion of *whose* fact it is. Given the source
"my brother is a lawyer" and the candidate fact "the user is a lawyer",
it returns 0.991 entailment: the professions match, so it entails.

That is the most dangerous failure mode for a memory gate. A wrong REJECT
gets flagged for a human. A wrong STORE enters memory silently, permanently,
and is recalled with full confidence forever.

This module runs *before* the NLI model and asks a narrower question: does
the source attribute this to the speaker, or to a third party?

Scope and limits — stated plainly:

    This is a heuristic over surface patterns, not a coreference resolver.
    It catches the common constructions ("my brother", "her neighbour's cat",
    "my colleague said", "she works as...") and will miss unusual phrasings.
    It is deliberately conservative: when third-party attribution is
    detected it *downgrades* a fact to REVIEW rather than rejecting it, so
    a false positive costs a human glance rather than lost information.

Fixed 2026 — two confirmed gaps, found via test_attribution_gaps.py:

    1. The possessive check only matched the literal word "my". "Her
       brother is a lawyer" produced third_party=False, confidence=high —
       the guard actively vouched the fact was safe when it wasn't. Now
       matches my/her/his/their/our.

    2. The third-person pronoun check (he/she/they/...) only ran inside
       the reporting-verb branch, so a plain declarative sentence like
       "She works as a lawyer" with no reporting verb never reached it at
       all. Now runs independently as its own check.

    Both produced the same dangerous silent-STORE failure mode this
    module exists to prevent — confidence=high, detected=[], "No
    third-party attribution detected" — on inputs that were, in fact,
    third-party attributions.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


# Relations that clearly denote a person other than the speaker.
# Deliberately excludes ambiguous ones like "my team" or "my company",
# where a fact about the group may legitimately be a fact about the user.
THIRD_PARTY_RELATIONS = [
    "brother", "sister", "sibling",
    "mother", "father", "mom", "dad", "parent", "parents",
    "son", "daughter", "child", "children", "kid", "kids",
    "wife", "husband", "spouse", "partner",
    "friend", "friends",
    "neighbour", "neighbor", "neighbours", "neighbors",
    "colleague", "colleagues", "coworker", "co-worker", "coworkers",
    "boss", "manager", "employee",
    "roommate", "flatmate", "housemate",
    "cousin", "uncle", "aunt", "nephew", "niece",
    "grandmother", "grandfather", "grandma", "grandpa",
    "doctor", "teacher", "landlord", "client", "customer",
]

# Possessive pronouns that mark the following relation as belonging to
# someone other than the speaker. "My" was the only one checked before —
# "her", "his", "their", "our" describe a third party's relation just as
# clearly and were previously invisible to this guard entirely.
THIRD_PARTY_POSSESSIVES = ["her", "his", "their", "our"]

# Verbs that mark reported speech — the source is relaying someone else's
# statement rather than making a first-person claim.
REPORTING_VERBS = [
    "said", "says", "told", "tells", "mentioned", "mentions",
    "explained", "explains", "claimed", "claims",
    "asked", "asks", "wrote", "writes",
]

# Third-person subject pronouns appearing as the actor of the fact.
THIRD_PERSON_SUBJECTS = ["he", "she", "they", "him", "her", "them"]


@dataclass
class AttributionResult:
    """Outcome of the attribution pre-check."""

    speaker_is_subject: bool     # False when the fact appears to belong to someone else
    confidence: str              # "high" | "low" — how sure the heuristic is
    detected: list[str]          # which patterns fired
    explanation: str

    @property
    def third_party(self) -> bool:
        return not self.speaker_is_subject


def _find_possessive_third_parties(text: str) -> list[str]:
    """
    Find '<possessive> <relation>' constructions, with or without a
    possessive 's.

    Covers "my brother", "her neighbour's", "their colleague" — any of
    my/her/his/their/our followed by a relation word. Previously only
    matched "my", which meant "her brother is a lawyer" produced no
    signal at all despite being exactly as clear a third-party
    attribution as "my brother is a lawyer".
    """
    found = []
    lowered = text.lower()
    all_possessives = ["my"] + THIRD_PARTY_POSSESSIVES
    for possessive in all_possessives:
        for relation in THIRD_PARTY_RELATIONS:
            pattern = rf"\b{re.escape(possessive)}\s+{re.escape(relation)}(?:'s|s')?\b"
            if re.search(pattern, lowered):
                found.append(f"{possessive} {relation}")
    return found


def _find_reported_speech(text: str) -> list[str]:
    """Find reporting verbs that signal the source is relaying someone else."""
    found = []
    lowered = text.lower()
    for verb in REPORTING_VERBS:
        if re.search(rf"\b{re.escape(verb)}\b", lowered):
            found.append(verb)
    return found


def _find_third_person_subject(text: str) -> list[str]:
    """
    Find third-person subject pronouns (he/she/they/...) anywhere in the
    source, independent of whether a reporting verb is present.

    Previously this check only ran inside the reported-speech branch, so
    a plain declarative sentence like "She works as a lawyer downtown" —
    no "my", no "said" — never reached any third-party check at all and
    fell straight through to speaker_is_subject=True, confidence=high.
    """
    found = []
    lowered = text.lower()
    for pronoun in THIRD_PERSON_SUBJECTS:
        if re.search(rf"\b{re.escape(pronoun)}\b", lowered):
            found.append(pronoun)
    return found


def split_sentences(text: str) -> list[str]:
    """
    Split source text into sentences on ./!/? boundaries.

    Deliberately simple — this exists to let a caller narrow attribution
    checking to the sentence most relevant to a specific fact, not to be a
    general-purpose sentence tokenizer. It will mishandle abbreviations
    ("Dr. Smith") and decimals, which is an acceptable tradeoff here: a
    slightly wrong split still narrows the search space, it just might not
    split at the exact grammatical boundary. Falls back to treating the
    whole text as one sentence if no boundary is found.
    """
    import re
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in pieces if p]


def _fact_subject_is_user(fact: str) -> bool:
    """
    Does the candidate fact claim something about the user specifically?

    Memory extractors conventionally phrase facts as "The user ...". If the
    fact is not about the user, attribution is not the relevant check.
    """
    lowered = fact.lower().strip()
    return lowered.startswith("the user") or lowered.startswith("user ")


def check_attribution(source: str, fact: str) -> AttributionResult:
    """
    Decide whether `fact` can be attributed to the speaker of `source`.

    Returns speaker_is_subject=False when the source appears to be describing
    a third party, which means a fact phrased as "the user ..." is probably
    a misattribution regardless of how strongly the NLI model entails it.
    """
    # If the fact isn't about the user, attribution isn't the question here.
    if not _fact_subject_is_user(fact):
        return AttributionResult(
            speaker_is_subject=True,
            confidence="low",
            detected=[],
            explanation="Fact is not phrased as a claim about the user; attribution not checked.",
        )

    possessives = _find_possessive_third_parties(source)
    reporting = _find_reported_speech(source)

    # Strongest signal: source is about "my/her/his/their/our <someone>",
    # fact is about the user.
    if possessives:
        return AttributionResult(
            speaker_is_subject=False,
            confidence="high",
            detected=possessives,
            explanation=(
                f"Source describes {possessives[0]}, not the speaker. "
                f"A fact about the user may be a misattribution."
            ),
        )

    # Reported speech: "my colleague mentioned she is moving" — the subject of
    # the reported clause is a third party.
    if reporting:
        third_person = _find_third_person_subject(source)
        if third_person:
            return AttributionResult(
                speaker_is_subject=False,
                confidence="high",
                detected=reporting + third_person,
                explanation=(
                    "Source reports what someone else said about a third party. "
                    "A fact about the user may be a misattribution."
                ),
            )
        return AttributionResult(
            speaker_is_subject=False,
            confidence="low",
            detected=reporting,
            explanation=(
                "Source contains reported speech; the fact may belong to "
                "the person being quoted rather than the speaker."
            ),
        )

    # Plain third-person subject with no possessive and no reporting verb —
    # e.g. "She works as a lawyer downtown." Checked independently now,
    # rather than only as a sub-check inside the reporting-verb branch.
    third_person = _find_third_person_subject(source)
    if third_person:
        return AttributionResult(
            speaker_is_subject=False,
            confidence="high",
            detected=third_person,
            explanation=(
                f"Source refers to a third party ({third_person[0]}), not "
                f"the speaker. A fact about the user may be a misattribution."
            ),
        )

    return AttributionResult(
        speaker_is_subject=True,
        confidence="high",
        detected=[],
        explanation="No third-party attribution detected in the source.",
    )