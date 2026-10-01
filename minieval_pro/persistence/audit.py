"""
Audit logging — an append-only, hash-chained record of every gate decision.

Why this exists
---------------
The gate itself is replicable: a memory engine that owns the write path could
add a faithfulness check in a few weeks. What is harder to bolt on afterwards
is a defensible record — one that can answer, months later, "what was decided
about this fact, under which rules, and why?"

That question is not academic. In a regulated deployment the operator has to
show that what the system stored about a person was justified at the time it
was stored. A score alone does not answer that. A record with the decision,
the evidence, and the policy in force does.

Design choices
--------------
Append-only by construction. Entries are written one JSON object per line to
a file opened in append mode. Nothing rewrites earlier lines. Previously this
was enforced by convention only — a determined operator could edit the file
and nothing would detect it.

Hash-chained (added after a real external project — Vestige, a signed
append-only memory log for AI agents — showed what "append-only" should
actually guarantee). Each entry now carries a hash of its own content plus
the previous entry's hash, so the log forms a chain: altering any entry
breaks every hash after it, and verify_chain() below catches that
deterministically rather than relying on the file format alone to imply
integrity.

Human-readable. JSONL survives without this library. If MiniEval disappears
tomorrow the log is still a text file anyone can read, grep, or load into
pandas. Binary or proprietary formats make an audit trail hostage to its tool.
The hash fields are visible in plain JSON too — verification doesn't require
this library either, just re-deriving a sha256 the same way.

Policy captured per entry. Each line records the policy name, version and
fingerprint that produced the decision. Change a threshold next month and the
old entries still say what was in force when they were written.

Fixed 2026 — summary() undercounted adjudicate() entries:

    stored/rejected/review only read the literal "STORE"/"REJECT"/"REVIEW"
    keys from by_verdict. adjudicate()'s ACCEPT and BLOCK verdicts were
    never included, so store_rate + reject_rate + review_rate silently
    failed to sum to 100% whenever the log contained any adjudicate()
    entries — the BLOCK/ACCEPT count existed in by_verdict but was
    invisible in the aggregate totals. Confirmed via a 4-entry test log
    (2 check(), 2 adjudicate()): rejected showed 1 instead of 2, and the
    missing BLOCK entry didn't appear in any of the three rate percentages.

    Fixed by folding ACCEPT into stored and BLOCK into rejected for these
    three aggregate counts only. by_verdict itself is untouched and still
    reports the real, distinct verdict strings — this only affects the
    three summary rates and totals.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union
import csv
import hashlib
import io
import json
import os


DEFAULT_LOG_NAME = "minieval_audit.jsonl"

# Fixed 64-character placeholder (matching sha256's hex digest length) used
# as the "previous hash" for the first entry in a chain. Distinguishable
# from any real hash (a real sha256 output is extremely unlikely to be all
# zeros), so a reader can tell "this is the genesis entry" from "this entry
# claims no predecessor but isn't actually first" at a glance.
GENESIS_HASH = "0" * 64

# Fields that are chain bookkeeping, not decision content. Excluded when
# computing an entry's own content hash — a field can't be part of its own
# hash input.
_HASH_METADATA_FIELDS = {"prev_hash", "entry_hash"}


def _default_log_path() -> Path:
    """
    Audit log location.

    Defaults to the working directory, not the package directory. A library
    that writes into its own install location surprises people and breaks on
    read-only installs.
    """
    override = os.environ.get("MINIEVAL_AUDIT_PATH")
    if override:
        return Path(override)
    return Path.cwd() / DEFAULT_LOG_NAME


def _content_hash(prev_hash: str, entry_content: dict) -> str:
    """
    sha256 of (prev_hash + canonical JSON of entry_content).

    Canonical means sort_keys=True and fixed separators — two dicts with
    the same keys and values always serialize identically regardless of
    insertion order, so the same logical entry always hashes the same way.
    """
    payload = prev_hash + json.dumps(
        entry_content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditLog:
    """
    Append-only, hash-chained JSONL record of gate decisions.

    Usage:
        log = AuditLog()                      # ./minieval_audit.jsonl
        log = AuditLog("logs/memory.jsonl")   # explicit path

        decision = gate.check(source, fact)
        log.record(decision)

        for entry in log.read():
            ...

        ok, report = log.verify_chain()
        log.to_csv("audit.csv")
    """

    def __init__(self, path: Optional[Union[str, Path]] = None):
        self.path = Path(path) if path else _default_log_path()

    # -- writing -----------------------------------------------------------

    def record(self, decision) -> dict:
        """
        Append one decision to the log, linked to the previous entry by hash.

        Accepts anything with a `to_dict()` method — GateDecision,
        AdjudicationDecision, or a caller's own record type — or a plain dict.
        Returns the entry as written, so a caller can inspect exactly what was
        persisted rather than assuming.
        """
        entry = decision.to_dict() if hasattr(decision, "to_dict") else dict(decision)

        # recorded_at is when the line was written; the decision's own
        # timestamp is when it was made. They differ if a caller batches
        # writes, and an auditor may care about both.
        entry["recorded_at"] = datetime.now(timezone.utc).isoformat()

        prev_hash = self._last_entry_hash()
        content = {k: v for k, v in entry.items() if k not in _HASH_METADATA_FIELDS}
        entry_hash = _content_hash(prev_hash, content)

        entry["prev_hash"] = prev_hash
        entry["entry_hash"] = entry_hash

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return entry

    def record_many(self, decisions: Iterable) -> int:
        """Append several decisions. Returns the count written."""
        count = 0
        for d in decisions:
            self.record(d)
            count += 1
        return count

    def _last_entry_hash(self) -> str:
        """
        The entry_hash of the most recently written entry, or GENESIS_HASH
        if the log is empty or its last entry predates hash-chaining.

        Reads the whole file's last entry rather than maintaining in-memory
        state, which keeps AuditLog instances stateless and safe to create
        fresh at any time — the cost is O(n) per write on a very large log,
        an honest tradeoff for a simple, always-correct implementation
        rather than a cache that could drift from the file on disk.
        """
        if not self.path.exists():
            return GENESIS_HASH

        last_entry = None
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last_entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

        if last_entry is None:
            return GENESIS_HASH

        return last_entry.get("entry_hash", GENESIS_HASH)

    # -- reading -----------------------------------------------------------

    def read(self) -> Iterator[dict]:
        """
        Yield entries in the order they were written.

        Malformed lines are skipped rather than raising. A corrupt line in a
        long-lived append-only file should not make the rest unreadable — the
        point of the format is that damage stays local.
        """
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def entries(self) -> list[dict]:
        """All entries as a list."""
        return list(self.read())

    def count(self) -> int:
        return sum(1 for _ in self.read())

    # -- integrity -----------------------------------------------------------

    def verify_chain(self) -> tuple[bool, dict]:
        """
        Walk the whole log and confirm every entry's hash actually matches
        its content plus the previous entry's hash.

        Returns (ok, report). report always has:
            "total"              entries examined
            "verified"           entries whose hash checked out AND are not
                                  downstream of any break
            "unhashed"           entries written before hash-chaining existed
                                  (no entry_hash field) — not a tamper signal,
                                  just a log that predates this feature
            "broken_at"          indices where the entry's OWN hash doesn't
                                  match its recomputed content+prev_hash —
                                  direct evidence that entry was altered
            "compromised"        indices that pass their own hash check but
                                  come after a broken entry, so cannot be
                                  trusted either — their prev_hash correctly
                                  points at the ORIGINAL (now-known-bad)
                                  upstream hash, which makes them internally
                                  self-consistent but still part of a chain
                                  with a known break in it

        ok is True only if both broken_at and compromised are empty.

        Why "compromised" exists as its own category, not folded into
        broken_at: a downstream entry that was never itself edited will
        correctly recompute its own hash against the (stale) hash its
        upstream neighbour still has on file — because a tamperer who only
        edits content and leaves the old hash in place doesn't touch
        anything downstream. That entry's own math is internally
        consistent. But "internally consistent" is not the same as
        "trustworthy": it's anchored to a chain link that is already known
        to be broken. Reporting it as merely "verified" would understate
        the damage; this field makes the honest claim explicit instead —
        confirmed via test_hash_chain.py, which tampers one entry and
        checks that every entry after it is flagged, not just the one
        directly edited.

        unhashed entries don't count as broken or compromised — a log that
        mixes a legacy unhashed prefix with a hashed suffix is not
        "tampered," it's a log where hashing was turned on partway through.
        That prefix simply can't be verified either way, and this reports
        that honestly rather than crashing on it or silently trusting it.
        """
        entries = self.entries()
        report = {
            "total": len(entries),
            "verified": 0,
            "unhashed": 0,
            "broken_at": [],
            "compromised": [],
        }

        expected_prev = GENESIS_HASH
        chain_started = False
        chain_compromised = False

        for i, entry in enumerate(entries):
            if "entry_hash" not in entry or "prev_hash" not in entry:
                report["unhashed"] += 1
                chain_started = False
                chain_compromised = False
                continue

            content = {k: v for k, v in entry.items() if k not in _HASH_METADATA_FIELDS}

            if not chain_started:
                expected_prev = entry["prev_hash"]
                chain_started = True

            recomputed = _content_hash(expected_prev, content)
            own_hash_ok = (
                entry["prev_hash"] == expected_prev
                and entry["entry_hash"] == recomputed
            )

            if not own_hash_ok:
                report["broken_at"].append(i)
                chain_compromised = True
            elif chain_compromised:
                report["compromised"].append(i)
            else:
                report["verified"] += 1

            # Advance using the entry's OWN stored hash (what downstream
            # entries were actually chained against at write time), not
            # the recomputed one — this is what makes a downstream entry's
            # self-consistency check meaningful at all, and is why
            # "compromised" has to be tracked as a separate signal rather
            # than expecting the raw hash math to propagate a break alone.
            expected_prev = entry["entry_hash"]

        ok = len(report["broken_at"]) == 0 and len(report["compromised"]) == 0
        return ok, report

    # -- export ------------------------------------------------------------

    def to_json(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Export as a single JSON array.

        Returns the JSON string; writes to `path` if given. Useful for handing
        a snapshot to someone who wants one file rather than a stream.
        """
        payload = json.dumps(self.entries(), indent=2, ensure_ascii=False)
        if path:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def to_csv(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Export as CSV.

        Columns are the union of all keys across entries, so a log containing
        both gate decisions and adjudications exports without losing fields.
        Missing values are left blank rather than dropped.
        """
        entries = self.entries()
        if not entries:
            return ""

        fieldnames: list[str] = []
        for entry in entries:
            for key in entry:
                if key not in fieldnames:
                    fieldnames.append(key)

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for entry in entries:
            writer.writerow({k: entry.get(k, "") for k in fieldnames})

        payload = buffer.getvalue()
        if path:
            Path(path).write_text(payload, encoding="utf-8", newline="")
        return payload

    # -- summary -----------------------------------------------------------

    def summary(self) -> dict:
        """
        Counts by verdict, plus which policies produced them.

        The policy breakdown matters: if a log spans a policy change, a single
        overall rate is misleading. Grouping by fingerprint keeps the two eras
        distinguishable.
        """
        entries = self.entries()
        by_verdict: dict[str, int] = {}
        by_policy: dict[str, dict] = {}

        for e in entries:
            verdict = e.get("verdict", "UNKNOWN")
            by_verdict[verdict] = by_verdict.get(verdict, 0) + 1

            fp = e.get("policy_fingerprint", "unknown")
            if fp not in by_policy:
                by_policy[fp] = {
                    "name": e.get("policy_name", "unknown"),
                    "version": e.get("policy_version", "unknown"),
                    "count": 0,
                    "verdicts": {},
                }
            by_policy[fp]["count"] += 1
            by_policy[fp]["verdicts"][verdict] = (
                by_policy[fp]["verdicts"].get(verdict, 0) + 1
            )

        total = len(entries)
        # ACCEPT (adjudicate's approve-overwrite) and BLOCK (adjudicate's
        # protect-existing) mean the same thing as STORE and REJECT for
        # these aggregate rates, but were previously invisible here — only
        # the literal "STORE"/"REJECT" keys were read from by_verdict. That
        # meant store_rate + reject_rate + review_rate silently failed to
        # sum to 100% whenever the log contained any adjudicate() entries,
        # since ACCEPT/BLOCK counts existed in by_verdict but were never
        # pulled into these totals. by_verdict itself is left untouched
        # below (it still reports the real, distinct verdict strings) —
        # only these three aggregate counts are normalised.
        stored = by_verdict.get("STORE", 0) + by_verdict.get("ACCEPT", 0)
        rejected = by_verdict.get("REJECT", 0) + by_verdict.get("BLOCK", 0)
        review = by_verdict.get("REVIEW", 0)

        return {
            "path": str(self.path),
            "total": total,
            "by_verdict": by_verdict,
            "stored": stored,
            "rejected": rejected,
            "flagged_for_review": review,
            "store_rate": round(100 * stored / total, 1) if total else 0.0,
            "reject_rate": round(100 * rejected / total, 1) if total else 0.0,
            "review_rate": round(100 * review / total, 1) if total else 0.0,
            "policies": by_policy,
        }