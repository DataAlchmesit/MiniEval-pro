# MiniEval — Design Document

**Status:** Frozen for v1.1
**Author:** Preeti Soni
**Last updated:** July 2026

This document is the working plan for MiniEval v1.1. It states the problem, the
position, the architecture, and the milestones. It is written before the code, and
the code is built against it. If a decision here turns out to be wrong, the
document changes first.

---

## 1 · The problem

AI memory systems extract facts from conversations and store them permanently.
Nobody checks whether those facts are true.

When a model extracts a memory, it can hallucinate. A conversation about a
*neighbour's* cat becomes "the user has a cat." A misheard sentence becomes a
stored fact. Once it is in the graph, it is recalled with total confidence, and it
quietly poisons every downstream response.

The second-order problem is worse. When a new fact conflicts with a stored one,
memory systems generally let the newer fact win. Recency is a reasonable default
until the newest fact is a hallucination — at which point the system silently
overwrites something that was true.

Neither problem is exotic. Both were reproduced against a live local memory engine
during development, with the audit log to show for it.

---

## 2 · The conceptual flip

> **Everyone is racing to make AI remember more. Nobody is asking whether what it
> remembers is true.**
>
> Memory systems optimise for *recall*. MiniEval optimises for *correctness*.
> A perfect memory of a false fact is worse than no memory at all.

The corollary, stated as a rule the system enforces:

> A new memory earns the right to overwrite an old one only if it is genuinely
> faithful to its own source. Recency and truth are not the same thing, and only
> one of them should get to overwrite a memory.

---

## 3 · Position and competitors

The pieces exist in the market. This specific combination does not.

| Category | Examples | What is missing |
|---|---|---|
| **Output validation** | Guardrails AI, TruLens | Validates at query-time. Does not gate what gets *written* into long-term memory. |
| **LLM observability** | Arize Phoenix | Observes drift and quality after the fact. Does not block a hallucination before it is stored. |
| **Memory engines** | Supermemory, Mem0, Zep | Conflict resolution lives inside the memory layer, tuned for recall quality. No faithfulness check before an overwrite is allowed. |
| **Memory benchmarks** | MemoryBench | Measures whether memory *recalls* correctly. Does not measure whether the stored fact was ever true. |

**The gap:** nobody combines *gate-before-write* with *evidence-based conflict
resolution* for AI memory.

### On defensibility — stated honestly

The gate itself is replicable. A memory engine owns the write path and could ship a
faithfulness flag in weeks. Historically, the layer that owns the write path absorbs
the layer that validates it.

The durable differentiator is therefore **not the scorer** — it is the **audit
trail**: reproducible decisions, append-only records, and policy versioning
("which rules were in effect on 4 March, and what did they decide?"). That is
unglamorous, compliance-grade rigour that a memory engine's roadmap is not pointed
at. Design effort should weight accordingly.

---

## 4 · Target user

**Buyer and user:** AI / platform engineering teams running a memory system in
production. They can `pip install` and evaluate in an afternoon. They already feel
the pain of garbage entering their memory graph.

**Reason they buy:** compliance pressure. In regulated environments the question
"prove what your AI stored about this person was accurate, and show me the record"
is not optional. Compliance creates the pressure; engineering does the installing.

**Non-negotiable constraint:** fully local operation. All three models run on-device
with no external API calls. A trust layer that phones home is useless to an
air-gapped or HIPAA-bound deployment. This is a hard requirement, not a feature.

---

## 5 · What the product is

**Today (v1.0):** a general LLM output scorer. Three strings in, three scores out.

**v1.1:** a **gate with a paper trail** — verify every fact before it is stored,
adjudicate conflicts by evidence, and produce a record that survives an audit.

Three capabilities that do not exist today:

1. **`MemoryGate`** — put in front of any memory system. `STORE` / `REVIEW` /
   `REJECT` with a reason, in a few lines instead of a hand-written loop.
2. **Overwrite adjudication** — `adjudicate(existing, incoming)`. The conceptual
   flip, shipped as an API.
3. **Audit trail as a first-class artifact** — every decision recorded with fact,
   source, score, verdict, reason, and the policy version in effect. One call
   produces a shareable report.

---

## 6 · Architecture

### Layering

```
minieval_pro/
├── core/          scorers + evaluator. Pure. No I/O, no DB, no network,
│                  no memory system. Fully testable offline.
├── gate/          MemoryGate + adjudicator. Decision logic only.
├── adapters/      memory-system connectors (Supermemory, Mem0, Zep, in-memory).
│                  Everything system-specific lives here and nowhere else.
└── persistence/   audit log writers (SQLite, JSON, CSV). Always optional.

dashboard/         FastAPI viewer + templates. NOT shipped to PyPI.
                   The library is the product; the web app is a local tool.
```

### The one rule

**`core/` never imports from `gate/`, `adapters/`, `persistence/`, or
`dashboard/`.** Dependencies point inward only.

This is not style preference. Three shipped bugs came directly from violating it:
a rogue `Evaluator` class defined inside `database/__init__.py`, a scoring function
that could not run without SQLite, and a database path that resolved into
`site-packages` on install. All three were architecture failures, not typos.

### Testability

`core/` must be runnable with no database, no network after first model download,
and no memory engine present. If a core test needs a running service, the layering
is wrong.

---

## 7 · Milestones

### M1 — Installable (complete)

The library was broken on fresh install. Nothing else mattered until `pip install`
worked on a clean machine.

- [x] Remove rogue `Evaluator` from `database/__init__.py`
- [x] Convert `evaluator.py` to relative imports
- [x] Make persistence opt-in (`Evaluator(save=False)` by default)
- [x] Auto-create the SQLite table instead of crashing
- [x] Move the DB path out of `site-packages` into the working directory
- [x] Verified: import and score both work from outside the project directory

### M2 — The gate

Memory becomes a first-class use case with its own interface, rather than a loop
each user rewrites.

- [x] `MemoryGate` class — three-bucket decision (`STORE` / `REVIEW` / `REJECT`)
- [x] `adjudicate()` — overwrite guarding by faithfulness
- [x] Adapter interface — works against Supermemory, Mem0, Zep, or a plain list
- [x] Reference adapter: Supermemory (the one already validated in the field)

### M3 — Scoring correctness

These are the failures found by *using* the tool, not by reading about it. They are
the difference between "works in a demo" and "works on someone else's data."

- [ ] **Separate `neutral` from `contradicts`.** Both currently score `0.00`, so an
      uncertain result is punished exactly as hard as a flat contradiction. This
      distorts ordinary results, not just edge cases. Highest priority in M3.
- [ ] **Inference gap.** "I moved to Bangalore" → "She lives in Bangalore" scores
      `neutral` at 98% confidence. A human reads it as plainly faithful.
- [ ] **Entity attribution.** "my neighbour's cat" → "the user has a cat" scored
      `0.96 faithful`. The model is keyword-matching, not resolving possession.
- [ ] Expose raw entailment / neutral / contradiction probabilities so callers can
      set their own thresholds instead of inheriting ours.
- [ ] Regression tests for every case above.

Revising M3 honestly
The priorities I wrote in DESIGN.md were wrong. Based on measurement rather than assumption:

Attribution (0/3, high confidence, silent bad writes) — most dangerous
Contradiction over-firing (unsupported → contradicts) — causes silent data loss
Neutral scoring near zero — real, but less urgent than I claimed

### M4 — Audit trail

The durable differentiator. Built as a product surface, not a debug log.

- [ ] Structured decision record: timestamp, fact, source, scores, verdict, reason
- [ ] **Policy versioning** — which thresholds and rules were in effect for a given
      decision, so past decisions remain reproducible after the policy changes
- [ ] Append-only semantics
- [ ] CSV and JSON export
- [ ] `generate_report()` — one call, shareable output
- [ ] Move `web/` out of the installable package into `dashboard/`

### M5 — Release

Documentation is not polish. An undocumented library has zero adopters regardless
of quality.

- [ ] README rewritten around the gate positioning
- [ ] Quickstart: install → gate a first memory in under two minutes
- [ ] One real worked example (Supermemory)
- [ ] `quiet=True` to silence model-loading output
- [ ] Version bump, changelog, publish to PyPI

---

## 8 · Tracked issues

Each is independently shippable.

| # | Issue | Milestone | Status |
|---|---|---|---|
| 1 | Rogue `Evaluator` in `database/__init__.py` | M1 | done |
| 2 | Old-name absolute imports in `evaluator.py` | M1 | done |
| 3 | `score()` requires a database | M1 | done |
| 4 | DB path resolves into `site-packages` | M1 | done |
| 5 | `MemoryGate` three-bucket API | M2 | open |
| 6 | `adjudicate()` overwrite guarding | M2 | open |
| 7 | Adapter interface + Supermemory adapter | M2 | open |
| 8 | `neutral` and `contradicts` both score 0.00 | M3 | open |
| 9 | Inference gap: "moved to X" → "lives in X" | M3 | open |
| 10 | Entity attribution: "neighbour's cat" → "user's cat" | M3 | open |
| 11 | Expose raw NLI probabilities | M3 | open |
| 12 | Structured audit record + policy versioning | M4 | open |
| 13 | CSV / JSON export + `generate_report()` | M4 | open |
| 14 | Move `web/` out of the installable package | M4 | open |
| 15 | `quiet=True` model loading | M5 | open |
| 16 | Docs, quickstart, worked example, release | M5 | open |

---

## 9 · Out of scope for v1.1

Named explicitly so they do not quietly creep in:

- Fine-tuning the NLI model. The M3 fixes are threshold, label, and framing work.
- Multi-language support.
- A hosted or SaaS version.
- Real-time streaming evaluation.
- Any memory engine of our own. MiniEval validates memory; it does not store it.

---

## 10 · Open questions

- Does the inference gap need a fine-tuned model, or can it be closed with better
  context framing? M3 should answer this empirically before any training work is
  considered.
- What is the right default policy for the `REVIEW` bucket in production — block,
  queue, or store-and-flag? Needs input from a real user, not a guess.
- Which memory engine gets the second adapter after Supermemory? Should be decided
  by whichever a real user is running, not by market share.
