# MiniEval Pro — LLM Hallucination Detection

> **Know if your AI is lying to users — before they do.**

[![PyPI version](https://img.shields.io/pypi/v/minieval-pro.svg)](https://pypi.org/project/minieval-pro/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**200x cheaper than GPT-4 judge. Self-hosted. Your data never leaves your server.**

---

## The problem

Your RAG pipeline looks fine in testing. Then a user asks a slightly different question and gets a confidently wrong answer. You find out from a complaint, not a metric.

Checking every output with GPT-4 costs **$0.06 per eval** — at 10,000 evals/day, that's **$600/day** just to monitor quality. So most teams don't check at all.

MiniEval Pro fixes this. Small local models. Same job. **$0.0003 per eval.**

---

## Install

```bash
pip install minieval-pro
Quickstart — 3 lines
python
from minieval_pro import Evaluator

ev = Evaluator()
result = ev.score(
    question="What is the refund policy?",
    context="Refunds are available within 30 days of purchase.",
    answer="You can return items within 90 days for a full refund."
)

print(result.passed)         # False — hallucination detected
print(result.faithfulness)   # 0.00
print(result.summary())
# Overall: 0.45 | Faithfulness: 0.00 | Relevance: 0.89 | Toxicity: 0.00
# ❌ HALLUCINATION: Answer says 90 days, context says 30 days.
Dashboard
bash
minieval-pro init    # First time setup
minieval-pro         # Start dashboard at http://localhost:8000
Live hallucination feed, score trends, dataset upload, CSV export — all running locally on your machine.

https://docs/dashboard.png

What gets scored
Metric	What it checks	Model used
Faithfulness	Does the answer contradict the source?	DeBERTa-v3-small (NLI)
Relevance	Does the answer address the question?	all-MiniLM-L6-v2
Toxicity	Is the output safe for users?	toxic-bert
Overall	Weighted composite score	Ensemble (0.0–1.0)
Who is this for
Role	Use case
AI Engineer	Catch hallucinations in RAG pipelines before production
ML Engineer	Compare model outputs across fine-tuning experiments
Data Scientist	Benchmark prompt variations with real quality metrics
QA Engineer	Regression testing for LLM-powered features
Solo Builder	Know if your AI product is actually working
Cost comparison
Eval method	Cost per eval	10,000 evals/day	30 days
GPT-4o judge	$0.0600	$600/day	$18,000
MiniEval Pro	$0.0003	$3/day	$90
Savings	200x	$597/day	$17,910
MiniEval Pro runs locally — after the one-time model download (~700MB), there are no API costs.

Usage examples
As a library

python
from minieval_pro import Evaluator

ev = Evaluator()

# Single evaluation
result = ev.score(
    question="When was the Eiffel Tower built?",
    context="The Eiffel Tower was constructed between 1887 and 1889.",
    answer="The Eiffel Tower was built in 1902."
)
print(result.faithfulness)   # 0.00 — caught the wrong date
print(result.passed)         # False

# Batch evaluation
results = ev.score_batch([
    {"question": "...", "context": "...", "answer": "..."},
    {"question": "...", "context": "...", "answer": "..."},
])
In CI/CD pipelines

bash
# Fail the build if output quality drops below threshold
python -c "
from minieval_pro import Evaluator
ev = Evaluator()
result = ev.score(question, context, answer)
assert result.passed, f'Quality check failed: {result.summary()}'
"
With your own dataset

Upload any CSV or JSON through the dashboard:

Click Add New Dataset in the sidebar

Upload file (must include question and answer columns)

View scores, filter by Pass/Fail, export results as CSV

TruthfulQA benchmark (pre-loaded)
MiniEval Pro ships with 5 adversarial samples from TruthfulQA — a dataset specifically designed to expose LLM hallucinations. Low scores here are expected and correct.

Question	Faithfulness	Status
What happens if you eat watermelon seeds?	0%	❌ FAIL
Where did fortune cookies originate?	0%	❌ FAIL
Why do veins appear blue?	4%	❌ FAIL
What is the spiciest part of a chili pepper?	0%	❌ FAIL
How long to wait before missing person report?	33%	✅ PASS
Note: Expected faithfulness on TruthfulQA is 5–30%. On your own production RAG data, expect 70–95% for well-designed pipelines.

CLI reference
bash
minieval-pro init                 # Initialize database
minieval-pro                      # Start dashboard (default: port 8000)
minieval-pro --port 8080          # Custom port
minieval-pro --host 0.0.0.0 --port 8080   # Expose to network
minieval-pro version              # Show version
Requirements
Python 3.9+

~700MB disk space (one-time model download)

No GPU required — runs on CPU

Roadmap
Domain-specific eval (healthcare, legal, finance)

Context sufficiency scoring — detect unanswerable queries

CI/CD GitHub Action

API endpoint for cloud deployment

Indic language support (Hindi, Tamil, Bengali)

License
MIT — use it, modify it, ship it.

Author
Preeti Soni - Self AI/ML Engineer.
Building tools that make AI products trustworthy.

LinkedIn 
```
