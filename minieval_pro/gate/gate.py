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
    """

    name: str = "default"
    version: str = "1.0"
    store_threshold: float = 0.5
    reject_on: tuple[str, ...] = ("contradicts",)
    overwrite_requires_faithful: bool = True

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
            f"(store>={self.store_threshold}, reject_on={list(self.reject_on)})"
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
    """

    def __init__(
        self,
        policy: Optional[Policy] = None,
        scorer: Optional[FaithfulnessScorer] = None,
    ):
        self.policy = policy or Policy()
        # Injectable for testing — a fake scorer keeps core tests offline.
        self._scorer = scorer or FaithfulnessScorer()

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _score(self, source: str, fact: str) -> FaithfulnessResult:
        return self._scorer.score(context=source, answer=fact)

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

        if result.label in self.policy.reject_on:
            verdict = REJECT
            reason = "Fact contradicts its source."
        elif result.label == "faithful" and result.score >= self.policy.store_threshold:
            verdict = STORE
            reason = "Fact is supported by its source."
        else:
            verdict = REVIEW
            reason = (
                "Fact is neither clearly supported nor contradicted. "
                "Flagged rather than guessed."
            )

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
        )