"""
Attribution hook — implements the pluggable contract described in
mem0ai/mem0#7283:

    hook(candidate, source_text, metadata) -> accepted / rejected-with-reason / uncertain

This is a thin translation layer over MemoryGate.check(), which already
implements this decision logic (tested, verified against real models) —
this module does not reimplement anything, it adapts the existing gate's
STORE/REJECT/REVIEW verdicts to the tri-state contract the hook interface
expects.

On `metadata`:
    Accepted but not yet used by the guard logic. This is reserved for
    role/actor_id, which @sattyamjjain's provenance analysis on the issue
    thread established is already a valid, currently-unused payload key
    on Mem0's inferred extraction path (see main.py, role and actor_id in
    _IDENTITY_KEYS). Speaker identity (who wrote this) is a different
    signal from subject identity (who the fact is about, which the
    attribution guard already checks via text). If Mem0 starts passing
    role/actor_id through metadata, this is the natural integration
    point — but that's future work, not implemented here. Passing it
    through unused now keeps the contract forward-compatible without
    pretending it does something it doesn't yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from minieval_pro.gate import MemoryGate, GateDecision


ACCEPTED = "accepted"
REJECTED = "rejected"
UNCERTAIN = "uncertain"

_VERDICT_MAP = {
    "STORE": ACCEPTED,
    "REJECT": REJECTED,
    "REVIEW": UNCERTAIN,
}


@dataclass
class HookVerdict:
    """
    The tri-state result of the attribution hook contract.

    verdict        "accepted" | "rejected" | "uncertain" — never a silent
                   pass; every candidate resolves to exactly one of these.
    reason         plain-English justification, present for all three
                   states (not just rejections) — an "uncertain" verdict
                   with no reason is not actionable for whoever routes it
                   to the lower-confidence tier.
    raw_decision   the underlying GateDecision, for callers who want the
                   full faithfulness score, label, and policy fingerprint
                   rather than just the tri-state summary. Not part of
                   the minimal contract, but costs nothing to expose.
    """

    verdict: str
    reason: str
    raw_decision: GateDecision


class AttributionHook:
    """
    Implements the (candidate, source_text, metadata) -> tri-state contract.

    Usage:
        hook = AttributionHook()
        result = hook.check(candidate="The user is a lawyer.",
                             source_text="My brother is a lawyer.",
                             metadata={"user_id": "alice"})
        print(result.verdict)  # "uncertain"
        print(result.reason)   # explanation
    """

    def __init__(self, gate: Optional[MemoryGate] = None):
        self.gate = gate or MemoryGate(quiet=True)

    def check(
        self,
        candidate: str,
        source_text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> HookVerdict:
        """
        candidate     the proposed fact, e.g. "The user is a lawyer."
        source_text   what the candidate should be verified against
        metadata      reserved for future use (see module docstring on
                       role/actor_id) — accepted, not currently read.
        """
        decision = self.gate.check(source=source_text, fact=candidate)

        verdict = _VERDICT_MAP.get(decision.verdict, UNCERTAIN)

        return HookVerdict(
            verdict=verdict,
            reason=decision.reason,
            raw_decision=decision,
        )