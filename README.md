# MiniEval - the trust layer for AI memory

> **Everyone is racing to make AI remember more. Nobody is asking whether what it remembers is true.**

[![PyPI version](https://img.shields.io/pypi/v/minieval-pro.svg)](https://pypi.org/project/minieval-pro/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Verify every fact before it enters your AI's memory. Adjudicate conflicts by
evidence rather than recency. Keep a record you can hand to an auditor.

Runs fully local. No API keys, no data leaving your machine.

---

## The one-sentence pitch

Memory systems optimise for **recall** - how reliably a stored fact comes back.
MiniEval optimises for **correctness** - whether the fact should have been stored
at all. A perfect memory of a false fact is worse than no memory.

---

## The problem

AI memory systems extract facts from conversations and store them permanently.
Nobody checks whether those facts are true.

A conversation about a *neighbour's* cat becomes "the user has a cat." A misheard
sentence becomes a permanent fact, recalled with total confidence, quietly
poisoning every downstream response.

The second-order problem is worse. When a new fact conflicts with a stored one,
memory systems generally let the newer fact win. Recency is a reasonable default
until the newest fact is a hallucination — at which point the system silently
overwrites something that was true.

```
Stored (true):    "The user is allergic to peanuts."
Incoming (false): "The user loves eating peanuts."
Recency wins:     the true memory is gone.
```

---

## Why this is differentiated

The pieces exist. This combination does not.

| Category | Examples | What's missing |
|---|---|---|
| **Output validation** | Guardrails AI, TruLens | Validates at query-time — not at the point a fact is *written* to memory |
| **LLM observability** | Arize Phoenix | Observes after the fact — doesn't block a hallucination before it's stored |
| **Memory engines** | Supermemory, Mem0, Zep | Conflict resolution tuned for recall quality; no faithfulness check before an overwrite |
| **Memory benchmarks** | MemoryBench | Measures whether memory *recalls* correctly — not whether the fact was ever true |

Nobody combines **gate-before-write** with **evidence-based conflict resolution**.

---

## Install

```bash
pip install minieval-pro
```

Python 3.10+. Roughly 700 MB of models download once, then run offline. No GPU
required.

---

## Quickstart

```python
from minieval_pro.gate import MemoryGate

gate = MemoryGate()

decision = gate.check(
    source="I am allergic to peanuts.",
    fact="The user loves eating peanuts.",
)

print(decision.verdict)   # REJECT
print(decision.reason)    # Fact contradicts its source.
```

Three outcomes, not two:

| Verdict | Meaning |
|---|---|
| `STORE` | Supported by the source - safe to remember |
| `REJECT` | Contradicts the source - a hallucination |
| `REVIEW` | Neither clearly supported nor contradicted - flagged for a human |

`REVIEW` is deliberate. A gate that forces every uncertain fact into yes/no will
either discard true information or admit false information. Flagging is the
honest third option.

---

## Guarding overwrites

The part nobody else does. A new memory earns the right to replace an old one
only if it is itself faithful to its own source.

```python
result = gate.adjudicate(
    existing_fact="The user is allergic to peanuts.",
    existing_source="I am allergic to peanuts.",
    incoming_fact="The user loves eating peanuts.",
    incoming_source="I had a great salad for lunch.",
)

print(result.verdict)   # BLOCK
print(result.reason)    # Incoming memory contradicts its own source.
                        # Existing memory protected.
```

A genuine update passes:

```python
gate.adjudicate(
    existing_fact="The user lives in Bangalore.",
    existing_source="I live in Bangalore.",
    incoming_fact="The user lives in Chennai.",
    incoming_source="I moved to Chennai last week.",
)
# ACCEPT — the new memory is faithful to its own source.
```

Each memory is scored against **its own** source, not against the other. Two
facts can both have been true at different moments; what matters is whether each
was justified when it was made.

---

## The audit trail

The durable part. Anyone can add a faithfulness check; what is hard to bolt on
later is a record that survives scrutiny months afterwards.

```python
from minieval_pro.persistence import AuditLog, generate_report

log = AuditLog("memory_audit.jsonl")

decision = gate.check(source, fact)
log.record(decision)

print(generate_report(log))
log.to_csv("audit.csv")
```

Every entry records the fact, its source, the verdict, the reason, the model's
raw probabilities, and the **policy fingerprint** that produced it:

```json
{"verdict": "REJECT", "fact": "The user loves eating peanuts.",
 "source": "I am allergic to peanuts.", "faithfulness": 0.0,
 "reason": "Fact contradicts its source.",
 "policy_fingerprint": "e7403d698e00", "policy_version": "1.0",
 "entailment": 0.0001, "contradiction": 0.999, "neutral": 0.001,
 "relatedness": 0.7865, "timestamp": "2026-07-19T14:31:17+00:00"}
```

Change a threshold next month and old entries still say what was in force when
they were written. That is what makes a decision reproducible rather than merely
recorded.

Append-only by construction — one JSON object per line, written in append mode.
Readable without this library. If MiniEval disappears, the log is still a text
file anyone can grep.

---

## Policies

Rules are a versioned, immutable object attached to every decision.

```python
from minieval_pro.gate import MemoryGate, Policy

policy = Policy(
    name="strict",
    version="2.0",
    store_threshold=0.7,      # higher bar to store
    min_relatedness=0.60,     # relatedness floor for contradictions
    check_attribution=True,   # third-party attribution guard
)

gate = MemoryGate(policy=policy)
```

---

## How it works

Three local models, and two guards around them.

| Component | Model | Role |
|---|---|---|
| Faithfulness | DeBERTa-v3-small (NLI) | Is the fact supported by, or contradicted by, its source? |
| Relevance | all-MiniLM-L6-v2 | Semantic similarity, used by the relatedness guard |
| Toxicity | toxic-bert | Output safety |

The guards exist because the NLI model answers questions it was not designed for:

**Attribution guard.** The model has no notion of *whose* fact this is. Given
"my brother is a lawyer" and "the user is a lawyer" it returns 0.99 entailment -
the professions match, so it entails. A pre-check detects third-party
attribution and downgrades to `REVIEW` rather than storing a misattribution.

**Relatedness guard.** NLI models have three labels and none of them means "these
texts are unrelated." Given an unrelated pair the model is forced to choose, and
often chooses contradiction - "I had a salad for lunch" versus "the user is a
doctor" returns contradiction at 0.985. When similarity falls below the policy
floor, the contradiction signal is treated as unreliable and the fact is flagged
rather than discarded.

---

## Measured results

18 labelled cases across five categories, scored on gate verdicts.

| Category | Before guards | After |
|---|---:|---:|
| Entailed - directly supported | 4/4 | 4/4 |
| Implied - supported after inference | 3/4 | 3/4 |
| Unsupported - unrelated, not contradictory | 1/3 | **3/3** |
| Contradicts - genuine conflicts | 4/4 | 4/4 |
| Attribution - belongs to a third party | 0/3 | **3/3** |
| **Overall** | **66.7%** | **94.4%** |

Reproduce it:

```bash
python -m minieval_pro.tests.run_gate_bench
```

The attribution row is the one that matters most. A wrong `REJECT` gets flagged
for a human; a wrong `STORE` enters memory silently and permanently. Those three
cases were confident false stores at 0.98–0.99 entailment.

---

## Dashboard

A local web view - audit history plus live checking.

```bash
pip install fastapi uvicorn
python dashboard/app.py
```

Open http://localhost:8000. Paste a source and a fact, watch the verdict appear
with its evidence, and see it land in the decision log.

Not shipped to PyPI. `pip install minieval-pro` gives you a library, not a web
server.

---

## Known limitations

Stated plainly, because a trust tool that hides its own failure modes is a
contradiction.

**One reasoning failure remains.** "I moved from Delhi to Bangalore last month"
versus "the user lives in Bangalore" returns contradiction at 0.774. The model
appears to read "moved *from* Delhi" as evidence against Bangalore. Neither guard
applies — the pair is highly related and correctly attributed. Fixing it likely
needs a different model or fine-tuning, both out of scope for this release.

**The guards are heuristics, not solvers.** Attribution detection matches surface
patterns — "my brother", "my colleague said". It will miss "the guy who lives
next door". It is deliberately conservative: when it fires it downgrades to
`REVIEW`, never `REJECT`, so a false positive costs a human glance rather than
lost information.

**The relatedness threshold comes from a small sample.** In a diagnostic set,
unrelated pairs scored 0.46–0.53 similarity while genuine contradictions scored
0.63–0.79. The default sits at 0.58, in that gap. Seven pairs is enough to
justify the approach, not enough to guarantee it generalises. Tune it against
your own data.

**`neutral` and `contradicts` both score near zero.** The gate distinguishes them
by label, so decisions are correct, but the numeric score is uninformative for
neutral results. Cosmetic rather than a correctness issue, and on the list.

---

## Project structure

```
minieval_pro/
├── scorers/       faithfulness, relevance, toxicity, attribution
├── gate/          MemoryGate, Policy, adjudication
├── persistence/   append-only audit log, reports, export
├── tests/         labelled test sets and benchmarks
└── evaluator.py   general-purpose scoring API

dashboard/         local web view — not shipped to PyPI
DESIGN.md          frozen design doc: problem, architecture, milestones
```

`scorers/` has no I/O and no knowledge of the layers above it. Dependencies point
inward. Three shipped bugs came from violating that rule, so it is now explicit.

---

## Also available: general-purpose scoring

The original evaluator API still exists for scoring any LLM output.

```python
from minieval_pro import Evaluator

ev = Evaluator()
result = ev.score(
    question="What is the refund policy?",
    context="Refunds are available within 30 days of purchase.",
    answer="You can return items within 90 days for a full refund.",
)
print(result.summary())
```

Persistence is opt-in — `Evaluator(save=True)` to write results to SQLite.
Scoring never requires a database.

---

## Roadmap

- Distinguish *unsupported* from *actively contradicting* in the score itself
- Adapters for specific memory engines, driven by what real users run
- Policy diffing - show what changed between two policy versions
- Entity-relationship awareness beyond surface patterns

---

## License

MIT — use it, modify it, ship it.

---

**Preeti Soni** — building tools that make AI systems trustworthy.
[LinkedIn](https://www.linkedin.com/in/preeti-soni-a5b8b6259/) · [GitHub](https://github.com/DataAlchmesit)
