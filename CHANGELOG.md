# Changelog

All notable changes to MiniEval are recorded here.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [1.1.0] — 2026-07-19

MiniEval changes shape in this release. 1.0 was a general-purpose scorer for
LLM outputs. 1.1 adds a **gate**: something you put in front of a memory system
to verify facts before they are stored, adjudicate conflicting memories by
evidence rather than recency, and keep a record that survives an audit.

The scoring API is unchanged and still supported.

### Added

**`MemoryGate` — verify facts before they enter memory**

Three outcomes rather than two. `STORE` when a fact is supported by its source,
`REJECT` when it contradicts, and `REVIEW` when it is neither. The third bucket
is deliberate: a gate that forces every uncertain fact into yes/no will either
discard true information or admit false information.

```python
from minieval_pro.gate import MemoryGate

gate = MemoryGate()
decision = gate.check(
    source="I am allergic to peanuts.",
    fact="The user loves eating peanuts.",
)
# REJECT — Fact contradicts its source.
```

**`adjudicate()` — guard overwrites**

When a new memory conflicts with a stored one, it may only replace it if it is
itself faithful to its own source. Recency is not evidence. A genuine update
("I moved to Chennai") passes; a hallucination does not, and the older true
memory is protected.

Each memory is scored against *its own* source rather than against the other.
Two facts can both have been true at different moments; what matters is whether
each was justified when it was made.

**`Policy` — versioned, immutable rules**

Thresholds and rules live in a frozen dataclass with a content hash. Every
decision records the policy name, version and fingerprint that produced it, so
a decision made under one set of rules stays interpretable after the rules
change.

**Attribution guard**

The NLI model has no notion of *whose* fact something is. Given "my brother is
a lawyer" and "the user is a lawyer" it returned 0.99 entailment — the
professions match, so it entailed. A pre-check now detects third-party
attribution and downgrades to `REVIEW` rather than storing a misattribution.

This was the most consequential fix in the release. A wrong `REJECT` gets
flagged for a human; a wrong `STORE` enters memory silently and permanently.

**Relatedness guard**

NLI models have three labels and none of them means "these texts are
unrelated." Given an unrelated pair the model is forced to choose, and often
chooses contradiction — "I had a salad for lunch" versus "the user is a doctor"
returned contradiction at 0.985. When semantic similarity falls below the
policy floor, the contradiction signal is treated as unreliable and the fact is
flagged rather than discarded.

**Append-only audit log**

`AuditLog` writes one JSON object per line in append mode. Every entry carries
the fact, its source, the verdict, the reason, the model's raw probabilities,
and the policy fingerprint. Readable without this library — if MiniEval
disappears, the log is still a text file anyone can grep.

Exports to CSV and JSON. `generate_report()` produces a text or markdown
summary with verdict counts, blocked entries with their evidence, the review
queue, and a per-policy breakdown.

**Raw NLI probabilities exposed**

`FaithfulnessResult` now carries `entailment`, `contradiction` and `neutral`
directly, so callers can set their own thresholds rather than inheriting ours.

**`quiet=True`**

Suppresses MiniEval's model-loading output. Available on `Evaluator`,
`MemoryGate`, and each scorer.

Note: HuggingFace's own progress bars are not affected and still appear.

**Local dashboard**

A web view of the audit log with live fact checking, in `dashboard/`. Not part
of the installed package — run it from a clone with
`pip install "minieval-pro[dashboard]"`.

### Fixed

**Fresh installs were broken.** Four separate bugs, each verified fixed:

- `database/__init__.py` contained a second, simplified `Evaluator` class that
  imported from the pre-rename package name. Any import touching the database
  module raised `ModuleNotFoundError`. Removed.
- `evaluator.py` used absolute imports against the old package name. Converted
  to relative imports, which survive renames.
- `score()` wrote to SQLite unconditionally, so scoring a single output failed
  with `no such table: evaluations` on a clean install. Persistence is now
  opt-in via `Evaluator(save=True)`; a score never requires a database.
- The database path resolved relative to the package, landing inside
  `site-packages` after installation. It now defaults to the working directory
  and honours `MINIEVAL_DB_PATH`.

**Dependencies were wrong.** The package declared FastAPI, uvicorn, Jinja2,
python-multipart and pandas — none of which the library imports — and omitted
`torch`, `transformers` and `sentence-transformers`, which it requires. A clean
install would not have worked. Corrected, with the web dependencies moved to a
`dashboard` extra.

### Changed

- Package no longer ships the FastAPI app, its templates, or a 6 MB benchmark
  dataset. Installed size is substantially smaller.
- `requires-python` raised to 3.10.
- README repositioned around memory trust rather than RAG evaluation.

### Removed

- `minieval_pro/web/` — superseded by `dashboard/`, which is not installed.
- `minieval-pro` CLI — it launched the removed web app. The dashboard runs with
  `python dashboard/app.py`.
- `database/update_database.py` — a migration script referencing tables and
  columns that no longer matched the schema.

### Measured

Gate accuracy over 18 labelled cases in five categories:

| Category | Before | After |
|---|---:|---:|
| Entailed | 4/4 | 4/4 |
| Implied | 3/4 | 3/4 |
| Unsupported | 1/3 | **3/3** |
| Contradicts | 4/4 | 4/4 |
| Attribution | 0/3 | **3/3** |
| **Overall** | **66.7%** | **94.4%** |

Reproduce with `python -m minieval_pro.tests.run_gate_bench`.

### Known limitations

- One reasoning failure remains. "I moved from Delhi to Bangalore last month"
  versus "the user lives in Bangalore" returns contradiction at 0.774. Neither
  guard applies — the pair is related and correctly attributed. Fixing it needs
  a different model or fine-tuning.
- Both guards are heuristics. Attribution detection matches surface patterns and
  will miss unusual phrasings. It fails safe: it downgrades to `REVIEW`, never
  `REJECT`.
- The relatedness threshold (0.58) comes from a seven-pair diagnostic. Enough to
  justify the approach, not enough to guarantee it generalises. Tune it against
  your own data.
- `neutral` and `contradicts` both produce a faithfulness score near zero. The
  gate distinguishes them by label so decisions are correct, but the numeric
  score is uninformative for neutral results.

---

## [1.0.0] — 2026-05

Initial release.

- `Evaluator` — weighted scoring across faithfulness, relevance and toxicity
- Faithfulness via DeBERTa-v3-small (NLI)
- Relevance via all-MiniLM-L6-v2
- Toxicity via toxic-bert
- Runs fully local, no API calls