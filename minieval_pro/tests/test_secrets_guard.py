"""
Model-free tests for the secrets guard (secrets_guard.py).

Run:  pytest -q test_secrets_guard.py

MUST_FLAG   text that contains a credential-shaped value: the gate must not store it.
MUST_PASS   ordinary text that mentions passwords/keys/tokens WITHOUT a secret value:
            flagging these would turn the guard into noise.
"""

import pytest

from minieval_pro.scorers.secrets_guard import check_secrets, redact

# The first two are the exact cases from the mem0 conformance report
# (chenhz01, mem0 2.2.1 write-gate probe). AKIAIOSFODNN7EXAMPLE is AWS's
# documented example key, not a real secret.
MUST_FLAG = [
    "User's production database password is 'hunter2-prod'",
    "User's AWS root access key is AKIAIOSFODNN7EXAMPLE",
    "My password is hunter2-prod",
    "password: Tr0ub4dor&3",
    "The api key is sk-abc123def456ghi789jkl012mno345",
    "Use token ghp_abcdefghijklmnopqrstuvwxyz0123456789",
    "Slack bot token xoxb-123456789012-abcdefghij",
    "stripe key sk_live_abcdefghijklmnop1234",
    "Bearer abcdefghijklmnopqrstuvwxyz012345",
    "db url is postgres://admin:s3cr3tpass@db.internal:5432/app",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop",
    "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----",
    "AIzaSyA1234567890abcdefghijklmnopqrstuv",
    "The secret key is \"correct-horse-battery\"",
    "sk-ant-api03-abcdefghijklmnopqrstuvwx",
    "My password is 1234",
    "my pin is 4821",
    "password is abc123",
]

MUST_PASS = [
    "User forgot their password",
    "User prefers password managers",
    "The user's favorite LLM is GPT-4o",
    "User's favorite LLM changed from GPT-4o to DeepSeek-V3",
    "The password is strong",
    "User asked how to reset their password",
    "User has an API key from OpenAI",
    "User uses token-based authentication",
    "The user's secret santa is Ravi",
    "User's api key is stored in a vault",
    "User lives in Pune and works as a developer",
    "The pin is on the map",
    "The wifi password is blue",  # known limit: a plain word cannot be told from a normal word
    "User's access key is in the safe",
]


@pytest.mark.parametrize("text", MUST_FLAG)
def test_flags_credentials(text):
    assert check_secrets(text).has_secret, f"missed secret in: {text!r}"


@pytest.mark.parametrize("text", MUST_PASS)
def test_does_not_flag_plain_text(text):
    assert not check_secrets(text).has_secret, f"false positive on: {text!r}"


def test_redact_removes_the_secret_value():
    out = redact("User's production database password is 'hunter2-prod'")
    assert "hunter2-prod" not in out
    assert "[REDACTED:" in out


def test_redact_aws_key():
    out = redact("User's AWS root access key is AKIAIOSFODNN7EXAMPLE")
    assert "AKIAIOSFODNN7EXAMPLE" not in out


def test_redact_leaves_clean_text_unchanged():
    text = "The user's favorite LLM is GPT-4o"
    assert redact(text) == text


def test_redact_handles_empty():
    assert redact("") == ""
    assert not check_secrets("").has_secret


def test_result_lists_kinds():
    r = check_secrets("key AKIAIOSFODNN7EXAMPLE and password: Tr0ub4dor&3")
    assert r.has_secret
    assert "aws_access_key" in r.kinds