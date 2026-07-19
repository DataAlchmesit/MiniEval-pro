"""
Run with: python tests/test_all.py
First run downloads models (~700MB total, one time only).
After that, each eval takes 1-3 seconds on CPU.
"""
from minieval import Evaluator

ev = Evaluator()

print("=" * 55)
print("TEST 1 — Good answer, should score HIGH")
print("=" * 55)
result = ev.score(
    question="What is the refund policy?",
    context="Customers can return items within 30 days of purchase for a full refund. Items must be unused.",
    answer="You can return unused items within 30 days for a full refund.",
)
print(result.summary())
print(f"Passed quality check: {result.passed()}")

print("\n" + "=" * 55)
print("TEST 2 — Hallucinated answer, should score LOW")
print("=" * 55)
result = ev.score(
    question="What is the refund policy?",
    context="Customers can return items within 30 days of purchase for a full refund.",
    answer="You can return items within 90 days for store credit.",
)
print(result.summary())
print(f"Passed quality check: {result.passed()}")

print("\n" + "=" * 55)
print("TEST 3 — Irrelevant answer, should score LOW")
print("=" * 55)
result = ev.score(
    question="What is the refund policy?",
    context="Customers can return items within 30 days.",
    answer="The weather in Mumbai is very hot in summer.",
)
print(result.summary())
print(f"Passed quality check: {result.passed()}")

print("\n" + "=" * 55)
print("TEST 4 — Real RAG scenario")
print("=" * 55)
result = ev.score(
    question="When was the Eiffel Tower built?",
    context="The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair.",
    answer="The Eiffel Tower was built between 1887 and 1889.",
)
print(result.summary())
print(f"Passed quality check: {result.passed()}")

print("\n" + "=" * 55)
print("TEST 5 — Batch scoring")
print("=" * 55)
batch = [
    {
        "question": "What is Python?",
        "context": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "answer": "Python is a programming language made by Guido van Rossum.",
    },
    {
        "question": "What is Python?",
        "context": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "answer": "Python is a type of snake found in tropical regions.",
    },
]
results = ev.score_batch(batch)
for i, r in enumerate(results, 1):
    print(f"Batch item {i}: overall={r.overall:.2f} passed={r.passed()}")