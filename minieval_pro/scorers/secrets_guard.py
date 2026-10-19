"""
Secrets checking — does this text contain a credential?

The NLI model checks whether a fact is *supported by its source*. A password
the user pasted is supported by the message that contains it, so a faithfulness
gate alone will happily STORE "The user's production database password is
'hunter2-prod'". Faithful is not the same as safe to remember.

This module runs *before* any storage decision and asks a narrower question:
does the text contain something shaped like a credential?

Scope and limits — stated plainly:

    This is pattern matching, not a secret scanner. It catches:
      - well-known token formats (AWS, GitHub, Slack, Stripe, Google, OpenAI-
        and Anthropic-style keys, JWTs, PEM private key blocks)
      - connection strings with an embedded password (scheme://user:pass@host)
      - "<password|secret|api key|token ...> is/=/: <value>" statements, where
        the value looks like a secret (contains a digit or symbol, or is quoted)

    It will miss credentials in unusual phrasings, and it will not catch
    other sensitive data (card numbers, national IDs, health details).
    A match is treated as "do not store"; a miss is NOT a guarantee that the
    text is safe.

    Deliberately conservative about plain words: "I forgot my password" and
    "the password is strong" are not flagged, because no secret value is present.

Why redact: a gate decision records the fact and its source, and the audit log
writes decisions to disk. If a secret is blocked from memory but copied
verbatim into the audit log, the leak has only moved. redact() replaces each
match with a labelled placeholder so decisions can be logged safely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# (kind, compiled pattern). Order matters only for redaction output labels.
_TOKEN_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("private_key", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----(?:.*?-----END [A-Z ]*PRIVATE KEY-----)?",
        re.DOTALL)),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[A-Z0-9]{16}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("stripe_key", re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}")),
    ("connection_string", re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s@/]+@[^\s/]+")),
]

# "<label> is/was/=/: <value>" — the value group is checked separately.
_LABELLED_VALUE = re.compile(
    r"\b(?P<label>(?:[a-z]+[ _-])?(?:password|passwd|pwd|passcode|pin|secret(?:[ _-]key)?|"
    r"api[ _-]?key|access[ _-]?key|secret[ _-]?access[ _-]?key|auth[ _-]?token|token|credentials?))"
    r"(?:\s*'s)?\s*(?:is|was|are|=|:)\s*(?P<value>\"[^\"]+\"|'[^']+'|\S+)",
    re.IGNORECASE,
)

_SYMBOLS = re.compile(r"[^A-Za-z\s]")


@dataclass
class SecretResult:
    has_secret: bool
    kinds: list[str] = field(default_factory=list)
    explanation: str = ""


def _looks_like_secret_value(raw: str) -> bool:
    """Quoted values count; unquoted ones need a digit or symbol (not a plain word)."""
    if len(raw) >= 2 and raw[0] in "\"'" and raw[-1] == raw[0]:
        inner = raw[1:-1]
        return len(inner) >= 4
    value = raw.rstrip(".,;)")
    if len(value) < 4:
        return False
    if any(ch.isdigit() for ch in value):
        return True  # "1234", "hunter2", "abc123": short digit-bearing values count
    return len(value) >= 6 and bool(_SYMBOLS.search(value))


def _find(text: str) -> list[tuple[str, int, int]]:
    """Return (kind, start, end) for each match, merged where they overlap."""
    hits: list[tuple[str, int, int]] = []
    for kind, pattern in _TOKEN_PATTERNS:
        for m in pattern.finditer(text):
            hits.append((kind, m.start(), m.end()))
    for m in _LABELLED_VALUE.finditer(text):
        if _looks_like_secret_value(m.group("value")):
            hits.append(("password_or_secret", m.start("value"), m.end("value")))
    hits.sort(key=lambda h: (h[1], -(h[2] - h[1])))
    merged: list[tuple[str, int, int]] = []
    for kind, s, e in hits:
        if merged and s < merged[-1][2]:
            continue  # overlaps a longer/earlier match; keep the first label
        merged.append((kind, s, e))
    return merged


def check_secrets(text: str) -> SecretResult:
    """Does `text` contain something shaped like a credential?"""
    if not text:
        return SecretResult(False)
    hits = _find(text)
    if not hits:
        return SecretResult(False)
    kinds = sorted({k for k, _, _ in hits})
    return SecretResult(
        True,
        kinds,
        "Contains credential-like content (" + ", ".join(kinds) + ").",
    )


def redact(text: str) -> str:
    """Replace each detected secret with [REDACTED:<kind>]."""
    if not text:
        return text
    hits = _find(text)
    if not hits:
        return text
    out, last = [], 0
    for kind, s, e in hits:
        out.append(text[last:s])
        out.append(f"[REDACTED:{kind}]")
        last = e
    out.append(text[last:])
    return "".join(out)