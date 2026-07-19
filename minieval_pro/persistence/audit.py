"""
Audit logging — an append-only record of every gate decision.

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
a file opened in append mode. Nothing rewrites earlier lines. This is not
enforced by permissions or checksums — a determined operator can edit the
file — but the format makes accidental mutation implausible and any edit
visible in a diff.

Human-readable. JSONL survives without this library. If MiniEval disappears
tomorrow the log is still a text file anyone can read, grep, or load into
pandas. Binary or proprietary formats make an audit trail hostage to its tool.

Policy captured per entry. Each line records the policy name, version and
fingerprint that produced the decision. Change a threshold next month and the
old entries still say what was in force when they were written.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union
import csv
import io
import json
import os


DEFAULT_LOG_NAME = "minieval_audit.jsonl"


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


class AuditLog:
    """
    Append-only JSONL record of gate decisions.

    Usage:
        log = AuditLog()                      # ./minieval_audit.jsonl
        log = AuditLog("logs/memory.jsonl")   # explicit path

        decision = gate.check(source, fact)
        log.record(decision)

        for entry in log.read():
            ...

        log.to_csv("audit.csv")
    """

    def __init__(self, path: Optional[Union[str, Path]] = None):
        self.path = Path(path) if path else _default_log_path()

    # -- writing -----------------------------------------------------------

    def record(self, decision) -> dict:
        """
        Append one decision to the log.

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
        stored = by_verdict.get("STORE", 0)
        rejected = by_verdict.get("REJECT", 0)
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