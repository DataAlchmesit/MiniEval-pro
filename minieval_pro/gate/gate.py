"""
MemoryGate — verify facts before they enter an AI's memory.

Memory systems optimise for recall. This gate optimises for correctness:
every candidate fact is checked against the source it came from, and a new
memory may only overwrite an existing one if it is genuinely more faithful.

Usage:
    from minieval_pro.gate import MemoryGate

    gate = MemoryGate()

    decision = gate.check(
        source="I moved from Delhi to Bangalore last month.",
        fact="The user lives in Delhi.",
    )
    print(decision.verdict)   # "REJECT"
    print(decision.reason)    # "Fact contradicts its source."
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional
import hashlib
import json

from ..scorers.faithfulness import FaithfulnessScorer, FaithfulnessResult
from ..scorers.relevance import RelevanceScorer
from ..scorers.attribution import check_attribution, split_sentences


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

STORE = "STORE"
REVIEW = "REVIEW"
REJECT = "REJECT"

ACCEPT_OVERWRITE = "ACCEPT"
BLOCK_OVERWRITE = "BLOCK"


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Policy:
    """
    The rules a gate applies, captured as an immutable, versioned object.

    A policy is attached to every decision the gate makes. This is what makes
    a decision reproducible after the rules change: an audit can ask "what was
    in effect on 4 March?" and get a real answer rather than today's settings.

    Fields:
        name              human-readable identifier for this policy
        version           bump when the rules change in a way that alters outcomes
        store_threshold   minimum faithfulness score to store a fact outright
        reject_on         labels that cause an outright rejection
        overwrite_requires_faithful
                          if True, an incoming memory may only overwrite an
                          existing one when it is itself labelled faithful
        min_relatedness   semantic similarity floor below which a "contradiction"
                          is treated as spurious. NLI models have no label for
                          "these texts are unrelated", so they often emit high
                          contradiction on pairs that simply have nothing to do
                          with each other. See note below on how this was chosen.
        check_attribution whether to run the third-party attribution pre-check

    On min_relatedness — an empirical default, not a truth:

        Measured over a small diagnostic set, unrelated pairs scored 0.46-0.53
        similarity while genuine contradictions scored 0.63-0.79. The default
        sits in that gap. The sample was small; treat this as a starting point
        and tune it against your own data.
    """

    name: str = "default"
    version: str = "1.0"
    store_threshold: float = 0.5
    reject_on: tuple[str, ...] = ("contradicts",)
    overwrite_requires_faithful: bool = True
    min_relatedness: float = 0.58
    check_attribution: bool = True

    def fingerprint(self) -> str:
        """
        Short, stable hash of the policy's contents.

        Two policies with identical rules produce the same fingerprint, so an
        audit log can group decisions made under the same rules even if the
        policy was renamed.
        """
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    def describe(self) -> str:
        return (
            f"{self.name} v{self.version} "
            f"(store>={self.store_threshold}, reject_on={list(self.reject_on)}, "
            f"min_relatedness={self.min_relatedness})"
        )


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------

@dataclass
class GateDecision:
    """One decision about one candidate fact."""

    verdict: str                    # STORE | REVIEW | REJECT
    fact: str                       # the candidate memory
    source: str                     # what it was checked against
    faithfulness: float             # 0.0 - 1.0
    label: str                      # faithful | neutral | contradicts
    reason: str                     # plain-English justification
    policy_name: str
    policy_version: str
    policy_fingerprint: str
    timestamp: str
    entailment: Optional[float] = None
    contradiction: Optional[float] = None
    neutral: Optional[float] = None
    relatedness: Optional[float] = None

    @property
    def stored(self) -> bool:
        return self.verdict == STORE

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"[{self.verdict}] {self.fact}\n"
            f"    faithfulness: {self.faithfulness:.2f} ({self.label})\n"
            f"    reason: {self.reason}"
        )


@dataclass
class AdjudicationDecision:
    """One decision about whether an incoming memory may replace an existing one."""

    verdict: str                    # ACCEPT | BLOCK | REVIEW
    existing_fact: str
    existing_faithfulness: float
    existing_label: str
    incoming_fact: str
    incoming_faithfulness: float
    incoming_label: str
    reason: str
    policy_name: str
    policy_version: str
    policy_fingerprint: str
    timestamp: str
    # When the existing memory was first written, if the caller knows it.
    # Optional, defaults to None, so existing callers and tests that don't
    # pass it keep working unchanged. Lets an auditor reading one decision
    # see both "when this overwrite happened" and "how old was the memory
    # that got replaced" without cross-referencing the log by fact text.
    existing_timestamp: Optional[str] = None
    # The original source text each fact was scored against. Optional and
    # defaults to None for the same backward-compatibility reason as
    # existing_timestamp. Without these, a report reading one
    # AdjudicationDecision can show the two competing facts but not what
    # either was actually checked against — a real gap found while fixing
    # the audit report, since the report has no way to show data the
    # decision itself never carried.
    existing_source: Optional[str] = None
    incoming_source: Optional[str] = None

    @property
    def overwrite_allowed(self) -> bool:
        return self.verdict == ACCEPT_OVERWRITE

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"[{self.verdict}] overwrite\n"
            f"    existing: {self.existing_fact} "
            f"({self.existing_faithfulness:.2f} {self.existing_label})\n"
            f"    incoming: {self.incoming_fact} "
            f"({self.incoming_faithfulness:.2f} {self.incoming_label})\n"
            f"    reason: {self.reason}"
        )


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

class MemoryGate:
    """
    A faithfulness gate for AI memory.

    The gate makes two kinds of decision:

      check()       should this candidate fact be stored at all?
      adjudicate()  may this incoming fact overwrite an existing memory?

    Both attach the policy in force to the decision, so past decisions stay
    reproducible after the policy changes.

    Two guards sit around the NLI scorer, because the model answers questions
    it was not designed for:

      attribution   the model has no notion of *whose* fact this is
      relatedness   the model has no label for "these texts are unrelated"

    Pass quiet=True to suppress model-loading output.
    """

    def __init__(
        self,
        policy: Optional[Policy] = None,
        scorer: Optional[FaithfulnessScorer] = None,
        relevance_scorer: Optional[RelevanceScorer] = None,
        quiet: bool = False,
    ):
        self.policy = policy or Policy()
        self.quiet = quiet
        # Injectable for testing — fakes keep core tests offline.
        self._scorer = scorer or FaithfulnessScorer(quiet=quiet)
        self._relevance = relevance_scorer or RelevanceScorer(quiet=quiet)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _score(self, source: str, fact: str) -> FaithfulnessResult:
        return self._scorer.score(context=source, answer=fact)

    def _relatedness(self, source: str, fact: str) -> float:
        """Semantic similarity between a source and a candidate fact."""
        return self._relevance.score(source, fact).score

    def _narrow_source_for_fact(self, source: str, fact: str) -> str:
        """
        Return the single sentence of `source` most relevant to `fact`.

        Used by both the attribution guard and the relatedness guard —
        not by the main faithfulness scoring, which still uses the full
        source unchanged, since broader context can legitimately matter
        for entailment reasoning.

        Why this exists: both guards were originally written assuming a
        source is about one thing. When a source is a whole multi-topic
        episode (the shape Mem0's real extraction pipeline produces —
        confirmed via their extracted_memories, which carries no per-fact
        source span), each guard fails the same way for the same reason:

          - Attribution: a third-party mention ANYWHERE in the episode
            wrongly flags every fact extracted from it, not just the one
            actually about that third party.
          - Relatedness: a fact clearly contradicted by ONE sentence in
            the episode can look "unrelated" when compared against the
            WHOLE episode, since most of the episode is about other
            things — downgrading a real REJECT to REVIEW for the wrong
            reason.

        Confirmed via test_attribution_narrowing.py: a 3-sentence source
        (Chennai / peanuts+salad / brother-is-a-lawyer) triggered both
        failure modes until this narrowing was applied to both guards.

        Falls back to the full source, unchanged, if there's only one
        sentence — this narrowing is an improvement on top of existing
        behaviour, not a requirement for it to keep working.
        """
        sentences = split_sentences(source)
        if len(sentences) <= 1:
            return source

        best_sentence = source
        best_score = -1.0
        for sentence in sentences:
            score = self._relatedness(sentence, fact)
            if score > best_score:
                best_score = score
                best_sentence = sentence

        return best_sentence

    def _decision(
        self,
        verdict: str,
        fact: str,
        source: str,
        result: FaithfulnessResult,
        reason: str,
        relatedness: Optional[float] = None,
    ) -> GateDecision:
        return GateDecision(
            verdict=verdict,
            fact=fact,
            source=source,
            faithfulness=result.score,
            label=result.label,
            reason=reason,
            policy_name=self.policy.name,
            policy_version=self.policy.version,
            policy_fingerprint=self.policy.fingerprint(),
            timestamp=self._now(),
            entailment=getattr(result, "entailment", None),
            contradiction=getattr(result, "contradiction", None),
            neutral=getattr(result, "neutral", None),
            relatedness=relatedness,
        )

    # -- public API --------------------------------------------------------

    def check(self, source: str, fact: str) -> GateDecision:
        """
        Decide whether a candidate fact should enter memory.

        STORE   the fact is supported by its source
        REJECT  the fact contradicts its source
        REVIEW  neither clearly supported nor contradicted — a human decides

        REVIEW exists deliberately. A gate that forces every uncertain fact
        into a yes/no will silently discard true information or admit false
        information; flagging is the honest third option.
        """
        result = self._score(source, fact)

        # Guard 1 — attribution.
        # The NLI model has no notion of whose fact this is: "my brother is a
        # lawyer" entails "the user is a lawyer" at 0.99. Downgrade rather
        # than storing a misattribution.
        #
        # Attribution is checked against the sentence most relevant to this
        # specific fact, not the full source — a multi-sentence source (the
        # normal shape for a whole conversation episode) can otherwise let a
        # third-party mention anywhere in it wrongly flag unrelated facts.
        # Faithfulness scoring above still uses the full source unchanged;
        # only this pattern-matching step is narrowed.
        if self.policy.check_attribution:
            attribution_source = self._narrow_source_for_fact(source, fact)
            attribution = check_attribution(attribution_source, fact)
            if attribution.third_party and attribution.confidence == "high":
                return self._decision(
                    verdict=REVIEW,
                    fact=fact,
                    source=source,
                    result=result,
                    reason=f"Possible misattribution. {attribution.explanation}",
                )

        # Guard 2 — relatedness.
        # The model has no label for "these texts are unrelated", so it often
        # emits high contradiction on pairs that simply have nothing to do
        # with each other. Rejecting those is silent data loss: an unsupported
        # fact should be flagged, not discarded.
        #
        # Relatedness is measured against the sentence most relevant to this
        # fact, not the full source — otherwise a fact clearly contradicted
        # by ONE sentence in a multi-topic source can look "unrelated" when
        # compared against the WHOLE source, wrongly downgrading a real
        # contradiction to REVIEW. See _narrow_source_for_fact for the full
        # reasoning; confirmed via test_attribution_narrowing.py.
        relatedness = None
        if result.label in self.policy.reject_on:
            relatedness_source = self._narrow_source_for_fact(source, fact)
            relatedness = self._relatedness(relatedness_source, fact)
            if relatedness < self.policy.min_relatedness:
                return self._decision(
                    verdict=REVIEW,
                    fact=fact,
                    source=source,
                    result=result,
                    reason=(
                        f"Source and fact appear unrelated "
                        f"(similarity {relatedness:.2f} < {self.policy.min_relatedness}). "
                        f"Treating the contradiction signal as unreliable; flagged "
                        f"rather than rejected."
                    ),
                    relatedness=relatedness,
                )

            return self._decision(
                verdict=REJECT,
                fact=fact,
                source=source,
                result=result,
                reason="Fact contradicts its source.",
                relatedness=relatedness,
            )

        if result.label == "faithful" and result.score >= self.policy.store_threshold:
            return self._decision(
                verdict=STORE,
                fact=fact,
                source=source,
                result=result,
                reason="Fact is supported by its source.",
            )

        return self._decision(
            verdict=REVIEW,
            fact=fact,
            source=source,
            result=result,
            reason=(
                "Fact is neither clearly supported nor contradicted. "
                "Flagged rather than guessed."
            ),
        )

    def check_many(self, source: str, facts: list[str]) -> list[GateDecision]:
        """Check several candidate facts extracted from the same source."""
        return [self.check(source, fact) for fact in facts]

    def adjudicate(
        self,
        existing_fact: str,
        existing_source: str,
        incoming_fact: str,
        incoming_source: str,
        existing_timestamp: Optional[str] = None,
    ) -> AdjudicationDecision:
        """
        Decide whether an incoming memory may overwrite an existing one.

        The rule: an incoming memory earns the right to overwrite only if it
        is itself faithful to its own source. Recency alone is not evidence.

        A legitimate update ("I moved to Chennai") is faithful to what the
        user said, so it is accepted. A hallucination ("the user loves
        peanuts", from a message about salad) is not, so the older true
        memory is protected.

        Note this compares each memory against *its own* source, not against
        each other. Two facts can both be true of different moments in time;
        what matters is whether each was justified when it was made.

        existing_timestamp is optional: if the caller knows when the existing
        memory was first written (e.g. looked up from the audit log), pass it
        so the returned decision lets an auditor see both when the overwrite
        happened and how old the memory being replaced was, without having to
        cross-reference the log by fact text. If omitted, it's simply None —
        current behavior for callers that don't pass it is unchanged.

        Known limitation: this does not compare existing_faithfulness against
        incoming_faithfulness. An incoming fact only needs to clear its own
        threshold independently — there is no check that it's at least as
        well-evidenced as what it would replace. In principle a weakly
        faithful incoming fact could overwrite a strongly faithful existing
        one. Several hand-constructed test cases attempted to reproduce this
        and did not land in the vulnerable score range — DeBERTa-v3-small
        tended toward extreme scores (near 0 or near 1) rather than the
        mid-range needed to trigger it — so this is a real gap by code
        inspection, not one confirmed in practice.

        Also does not run the relatedness guard used in check(): an incoming
        fact that is simply unrelated to its own source (rather than a true
        contradiction of it) can still be labelled "contradicts" by the NLI
        model and BLOCK here. In testing this produced the correct outcome
        by coincidence — protecting the existing memory — but for a reason
        that wasn't actually verified.
        """
        existing = self._score(existing_source, existing_fact)
        incoming = self._score(incoming_source, incoming_fact)

        if incoming.label in self.policy.reject_on:
            verdict = BLOCK_OVERWRITE
            reason = (
                "Incoming memory contradicts its own source. "
                "Existing memory protected."
            )
        elif (
            incoming.label == "faithful"
            and incoming.score >= self.policy.store_threshold
        ):
            verdict = ACCEPT_OVERWRITE
            reason = "Incoming memory is faithful to its source — legitimate update."
        elif not self.policy.overwrite_requires_faithful:
            verdict = ACCEPT_OVERWRITE
            reason = "Policy does not require faithfulness for an overwrite."
        else:
            verdict = REVIEW
            reason = (
                "Incoming memory is not clearly faithful. "
                "Not confident enough to overwrite an existing memory."
            )

        return AdjudicationDecision(
            verdict=verdict,
            existing_fact=existing_fact,
            existing_faithfulness=existing.score,
            existing_label=existing.label,
            incoming_fact=incoming_fact,
            incoming_faithfulness=incoming.score,
            incoming_label=incoming.label,
            reason=reason,
            policy_name=self.policy.name,
            policy_version=self.policy.version,
            policy_fingerprint=self.policy.fingerprint(),
            timestamp=self._now(),
            existing_timestamp=existing_timestamp,
            existing_source=existing_source,
            incoming_source=incoming_source,
        )