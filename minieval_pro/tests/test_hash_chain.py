"""
Tests AuditLog's hash chain: writes real entries, confirms verify_chain()
passes clean, then actually tampers with the file on disk and confirms
tampering -- and its downstream effect -- is detected.

Run with:
    pytest minieval_pro/tests/test_hash_chain.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest

from minieval_pro.persistence.audit import AuditLog, GENESIS_HASH


def make_decision(verdict, fact):
    return {
        "verdict": verdict,
        "fact": fact,
        "source": f"source for {fact}",
        "faithfulness": 0.9,
        "label": "faithful",
        "reason": "test entry",
        "policy_name": "default",
        "policy_version": "1.0",
        "policy_fingerprint": "abc123",
        "timestamp": "2026-10-01T00:00:00+00:00",
    }


@pytest.fixture
def three_entry_log():
    """A fresh log with three real recorded entries, cleaned up after."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    tmp.close()
    log = AuditLog(tmp.name)

    log.record(make_decision("STORE", "fact one"))
    log.record(make_decision("REJECT", "fact two"))
    log.record(make_decision("REVIEW", "fact three"))

    yield log

    Path(log.path).unlink(missing_ok=True)


def test_clean_chain_verifies_ok(three_entry_log):
    ok, report = three_entry_log.verify_chain()

    assert ok is True
    assert report["verified"] == 3
    assert report["broken_at"] == []
    assert report["compromised"] == []


def test_genesis_entry_uses_genesis_hash(three_entry_log):
    entries = three_entry_log.entries()
    assert entries[0]["prev_hash"] == GENESIS_HASH


def test_tampering_is_detected_at_the_correct_index(three_entry_log):
    """
    Edit entry 1's content on disk, deliberately leaving its stored
    entry_hash untouched -- exactly what a naive tamper attempt looks
    like -- and confirm verify_chain() catches it.
    """
    with open(three_entry_log.path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    tampered = json.loads(lines[1])
    tampered["fact"] = "SOMEONE EDITED THIS FACT AFTER THE FACT"
    lines[1] = json.dumps(tampered) + "\n"

    with open(three_entry_log.path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    ok, report = three_entry_log.verify_chain()

    assert ok is False
    assert report["broken_at"] == [1]


def test_downstream_entry_is_flagged_compromised_not_silently_valid(three_entry_log):
    """
    Entry 2 is never itself edited, so its own hash math is internally
    consistent -- it correctly matches entry 1's stale, pre-tamper hash.
    That's exactly why it needs its own category: not "broken" (nobody
    touched its content), but "compromised" (anchored to a chain link
    now known to be bad). An earlier version of verify_chain() missed
    this entirely and reported entry 2 as "verified" -- this test is
    what caught that.
    """
    with open(three_entry_log.path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    tampered = json.loads(lines[1])
    tampered["fact"] = "SOMEONE EDITED THIS FACT AFTER THE FACT"
    lines[1] = json.dumps(tampered) + "\n"

    with open(three_entry_log.path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    ok, report = three_entry_log.verify_chain()

    assert ok is False
    assert 2 in report["compromised"]
    assert 2 not in report["broken_at"]  # entry 2 itself was never edited