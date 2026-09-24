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
    It catches the common constructions ("my brother", "her neighbour's
    cat", "my colleague said", "she works as...") and will miss unusual
    phrasings. It is deliberately conservative: when third-party
    attribution is detected it *downgrades* a fact to REVIEW rather than
    rejecting it, so a false positive costs a human glance rather than
    lost information.

Fix history:

    1. Possessive check originally only matched "my". Widened to
       my/her/his/their/our — "her brother is a lawyer" previously
       produced third_party=False, confidence=high (a silent false-STORE
       risk), now correctly caught.

    2. Third-person pronoun check originally only ran inside the
       reported-speech branch, missing plain declarative sentences like
       "she works as a lawyer downtown" with no reporting verb. Added as
       an independent standalone check.

    3. The standalone check from (2) is unconditional — it flags a
       third-person pronoun anywhere in the source, regardless of
       grammatical role. This caused a real false positive, found via
       the attribution regression corpus (neg_002): "I told her about
       the new apartment I found" was flagged, because "her" appears in
       the source — even though "her" is the OBJECT of "told" (someone
       the user spoke TO), not the subject of any fact. Fixed by
       excluding a pronoun when it immediately follows a small set of
       common verbs that take a person as their object rather than
       their subject (OBJECT_POSITION_VERBS below). This is a targeted
       heuristic improvement, not a grammatical parse — it narrows the
       false-positive class demonstrated by neg_002, it does not
       eliminate every possible false positive of this shape.
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
    "teammate",
]

# Possessive pronouns that mark the following relation as belonging to
# someone other than the speaker. "My" was the only one checked
# originally — "her", "his", "their", "our" describe a third party's
# relation just as clearly.
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

# Common verbs that typically take a person as their OBJECT, not subject
# — "I told her", "I asked him", "I invited them". A third-person pronoun
# immediately following one of these is very likely the object of the
# sentence (someone the speaker addressed or interacted with), not the
# subject of a fact being reported. This list is deliberately small and
# common-case; it will not catch every object-position construction, the
# same tradeoff as every other heuristic in this module.
OBJECT_POSITION_VERBS = [
    "told", "asked", "informed", "showed", "gave", "called",
    "invited", "helped", "thanked", "visited", "met", "saw",
]


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
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in pieces if p]


def _find_possessive_third_parties(text: str) -> list[str]:
    """
    Find '<possessive> <relation>' constructions, with or without a
    possessive 's.

    Covers "my brother", "her neighbour's", "their colleague" — any of
    my/her/his/their/our followed by a relation word.
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


def _find_third_person_subject(text: str, exclude_object_position: bool = False) -> list[str]:
    """
    Find third-person subject pronouns (he/she/they/...) in the source.

    exclude_object_position: when True, a pronoun immediately preceded by
    one of OBJECT_POSITION_VERBS ("told her", "asked him") is NOT counted
    — it's very likely the object of the sentence, not the subject of a
    fact. Used by the standalone fallback check in check_attribution(),
    which previously flagged ANY pronoun occurrence regardless of role
    and produced a real false positive (see fix history above). Left as
    False by default so the reported-speech branch's internal use of this
    function (a different, lower-risk context) is unchanged.
    """
    found = []
    lowered = text.lower()
    for pronoun in THIRD_PERSON_SUBJECTS:
        for match in re.finditer(rf"\b{re.escape(pronoun)}\b", lowered):
            if exclude_object_position:
                preceding_text = lowered[:match.start()]
                preceding_words = preceding_text.split()
                if preceding_words and preceding_words[-1] in OBJECT_POSITION_VERBS:
                    continue
            found.append(pronoun)
            break  # one hit per pronoun is enough to know it's present
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
        third_person = _find_third_person_subject(source, exclude_object_position=True)
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
    # e.g. "She works as a lawyer downtown." Checked independently, with
    # object-position pronouns excluded — "I told her about..." should NOT
    # trigger this, since "her" there is the object of "told", not the
    # subject of a fact. See fix history (3) above.
    third_person = _find_third_person_subject(source, exclude_object_position=True)
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