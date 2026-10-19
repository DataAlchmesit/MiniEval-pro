"""
Gate-level tests for the secrets guard, run offline with fake scorers.

The fake faithfulness scorer says EVERYTHING is faithful. That is the point:
a pasted password is "faithful" to the message that contains it, so without the
secrets guard the gate would STORE it.

Run:  pytest -q test_gate_secrets.py
"""
from types import SimpleNamespace

from minieval_pro.gate import MemoryGate, Policy


class AlwaysFaithful:
    def score(self, context, answer):
        return SimpleNamespace(
            score=0.99, label="faithful",
            entailment=0.99, contradiction=0.0, neutral=0.01,
        )


class AlwaysRelated:
    def score(self, a, b):
        return SimpleNamespace(score=0.9)


def make_gate(**policy_kwargs):
    return MemoryGate(
        policy=Policy(**policy_kwargs),
        scorer=AlwaysFaithful(),
        relevance_scorer=AlwaysRelated(),
        quiet=True,
    )


PW_SOURCE = "My production database password is 'hunter2-prod'. Please remember it."
PW_FACT = "The user's production database password is 'hunter2-prod'."
AWS_SOURCE = "My AWS root access key is AKIAIOSFODNN7EXAMPLE."
AWS_FACT = "The user's AWS root access key is AKIAIOSFODNN7EXAMPLE."


def test_without_guard_the_password_is_stored():
    # Documents the original gap: faithful != safe to remember.
    d = make_gate(check_secrets=False).check(PW_SOURCE, PW_FACT)
    assert d.verdict == "STORE"


def test_password_fact_is_rejected():
    d = make_gate().check(PW_SOURCE, PW_FACT)
    assert d.verdict == "REJECT"
    assert "Not stored" in d.reason


def test_aws_key_fact_is_rejected():
    d = make_gate().check(AWS_SOURCE, AWS_FACT)
    assert d.verdict == "REJECT"


def test_decision_does_not_contain_the_secret():
    d = make_gate().check(PW_SOURCE, PW_FACT)
    blob = repr(d.to_dict())
    assert "hunter2-prod" not in blob


def test_secret_only_in_source_is_redacted_but_clean_fact_still_stored():
    d = make_gate().check(
        "My password is hunter2-prod. I also live in Pune.",
        "The user lives in Pune.",
    )
    assert d.verdict == "STORE"
    assert "hunter2-prod" not in repr(d.to_dict())


def test_ordinary_fact_is_untouched():
    d = make_gate().check("I am allergic to peanuts.", "The user is allergic to peanuts.")
    assert d.verdict == "STORE"
    assert d.fact == "The user is allergic to peanuts."
    assert d.source == "I am allergic to peanuts."


def test_check_many_applies_the_guard_to_each_fact():
    decisions = make_gate().check_many(
        "My password is hunter2-prod. I live in Pune.",
        ["The user's password is hunter2-prod.", "The user lives in Pune."],
    )
    assert [d.verdict for d in decisions] == ["REJECT", "STORE"]


def test_adjudicate_blocks_overwrite_with_a_secret_and_redacts():
    d = make_gate().adjudicate(
        existing_fact="The user lives in Pune.",
        existing_source="I live in Pune.",
        incoming_fact="The user's password is hunter2-prod.",
        incoming_source="My password is hunter2-prod.",
    )
    assert d.verdict == "BLOCK"
    assert "hunter2-prod" not in repr(d.to_dict())


def test_adjudicate_ordinary_update_still_accepted():
    d = make_gate().adjudicate(
        existing_fact="The user lives in Pune.",
        existing_source="I live in Pune.",
        incoming_fact="The user lives in Chennai.",
        incoming_source="I moved to Chennai last week.",
    )
    assert d.verdict == "ACCEPT"


def test_policy_fingerprint_changes_when_guard_is_toggled():
    assert Policy(check_secrets=True).fingerprint() != Policy(check_secrets=False).fingerprint()