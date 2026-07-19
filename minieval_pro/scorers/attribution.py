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
    It catches the common constructions ("my brother", "my neighbour's cat",
    "my colleague said") and will miss unusual phrasings. It is deliberately
    conservative: when third-party attribution is detected it *downgrades*
    a fact to REVIEW rather than rejecting it, so a false positive costs a
    human glance rather than lost information.
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
    """Find 'my <relation>' constructions, with or without a possessive 's."""
    found = []
    lowered = text.lower()
    for relation in THIRD_PARTY_RELATIONS:
        # "my brother", "my neighbour's", "my colleague"
        pattern = rf"\bmy\s+{re.escape(relation)}(?:'s|s')?\b"
        if re.search(pattern, lowered):
            found.append(f"my {relation}")
    return found


def _find_reported_speech(text: str) -> list[str]:
    """Find reporting verbs that signal the source is relaying someone else."""
    found = []
    lowered = text.lower()
    for verb in REPORTING_VERBS:
        if re.search(rf"\b{re.escape(verb)}\b", lowered):
            found.append(verb)
    return found


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

    # Strongest signal: source is about "my <someone>", fact is about the user.
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
        lowered = source.lower()
        has_third_person = any(
            re.search(rf"\b{p}\b", lowered) for p in THIRD_PERSON_SUBJECTS
        )
        if has_third_person:
            return AttributionResult(
                speaker_is_subject=False,
                confidence="high",
                detected=reporting + ["third-person subject"],
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

    return AttributionResult(
        speaker_is_subject=True,
        confidence="high",
        detected=[],
        explanation="No third-party attribution detected in the source.",
    )
