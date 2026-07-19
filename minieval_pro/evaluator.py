from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time
import re

from .scorers.faithfulness import FaithfulnessScorer, FaithfulnessResult
from .scorers.relevance import RelevanceScorer, RelevanceResult
from .scorers.toxicity import ToxicityScorer, ToxicityResult

from .database.db import save_evaluation


@dataclass
class EvalResult:
    """Complete evaluation result for one LLM output."""
    overall: float              # 0.0 to 1.0 — the single headline score

    faithfulness: FaithfulnessResult
    relevance: RelevanceResult
    toxicity: ToxicityResult

    # Weights used to calculate overall score
    weights: dict = field(default_factory=lambda: {
        "faithfulness": 0.5,
        "relevance": 0.3,
        "safety": 0.2,
    })

    def summary(self) -> str:
        """Human-readable summary of the evaluation."""
        lines = [
            f"Overall Score:   {self.overall:.2f} / 1.00",
            f"Faithfulness:    {self.faithfulness.score:.2f} ({self.faithfulness.label})",
            f"Relevance:       {self.relevance.score:.2f} ({self.relevance.label})",
            f"Toxicity:        {self.toxicity.score:.2f} ({self.toxicity.label})",
            "",
            "Details:",
            f"  {self.faithfulness.explanation}",
            f"  {self.relevance.explanation}",
            f"  {self.toxicity.explanation}",
        ]
        return "\n".join(lines)

    def passed(self, threshold: float = 0.6) -> bool:
        """Quick check — did this output meet minimum quality?"""
        return self.overall >= threshold and not self.toxicity.is_toxic


class Evaluator:
    """
    Main MiniEval API. This is what users import and use.

    Usage:
        from minieval_pro import Evaluator

        ev = Evaluator()
        result = ev.score(
            question = "What is the capital of France?",
            context  = "France is a country in Europe. Its capital is Paris.",
            answer   = "The capital of France is Paris.",
        )
        print(result.summary())
        print(result.passed())   # True

    To persist every evaluation to a local SQLite database:

        ev = Evaluator(save=True)
    """

    def __init__(
        self,
        faithfulness_weight: float = 0.5,
        relevance_weight: float = 0.3,
        safety_weight: float = 0.2,
        save: bool = False,
    ):
        # Validate weights sum to 1.0
        total = faithfulness_weight + relevance_weight + safety_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.2f}. "
                f"Adjust faithfulness_weight, relevance_weight, safety_weight."
            )

        self.fw = faithfulness_weight
        self.rw = relevance_weight
        self.sw = safety_weight

        # Persistence is opt-in. A score should never require a database.
        self.save = save

        # Create scorer instances — models load lazily on first use
        self._faithfulness = FaithfulnessScorer()
        self._relevance    = RelevanceScorer()
        self._toxicity     = ToxicityScorer()

    def _extract_contradiction(self, context: str, answer: str) -> str | None:
        """Try to find what exactly contradicts between context and answer"""
        # Check for number differences
        context_numbers = re.findall(r'\d+', context)
        answer_numbers = re.findall(r'\d+', answer)

        if context_numbers and answer_numbers and context_numbers[0] != answer_numbers[0]:
            return f"Your source says {context_numbers[0]}, but the answer says {answer_numbers[0]}."

        # Check for percentage differences
        context_percent = re.findall(r'(\d+)%', context)
        answer_percent = re.findall(r'(\d+)%', answer)

        if context_percent and answer_percent and context_percent[0] != answer_percent[0]:
            return f"Your source says {context_percent[0]}%, but the answer says {answer_percent[0]}%."

        return None

    def _build_failure_reason(
        self,
        question: str,
        context: str,
        answer: str,
        faith_result,
        rel_result,
        tox_result,
        overall: float,
        passed: bool,
        threshold: float,
    ) -> str:
        """Plain-English explanation of why an evaluation passed or failed."""
        if passed:
            return (
                f"PASSED: Quality score {overall:.0%} meets threshold "
                f"({threshold:.0%}). Answer is faithful, relevant, and safe."
            )

        reasons = []

        # Faithfulness / hallucination check
        if faith_result.score < 0.5:
            contradiction = self._extract_contradiction(context, answer)
            if contradiction:
                reasons.append(f"HALLUCINATION: {contradiction}")
            else:
                context_preview = context[:150] if context else "your source document"
                answer_preview = answer[:150]
                reasons.append(f'HALLUCINATION: The answer says "{answer_preview}..."')
                reasons.append(f"   This is not supported by: {context_preview}...")
            reasons.append(
                f"   Faithfulness confidence: {faith_result.score:.0%}"
            )

        # Relevance check
        if rel_result.score < 0.5:
            reasons.append("IRRELEVANT ANSWER: The answer doesn't address the question.")
            reasons.append(f'   Question: "{question[:100]}..."')
            reasons.append(f'   Answer: "{answer[:100]}..."')
            reasons.append(f"   Relevance score: {rel_result.score:.0%} (needs >50%)")

        # Toxicity check
        if tox_result.is_toxic:
            reasons.append("TOXIC CONTENT: The answer contains unsafe language.")
            reasons.append(f'   Offending text: "{answer[:150]}..."')
            reasons.append(f"   Toxicity score: {tox_result.score:.0%} (should be <30%)")

        # Low overall score with no single specific cause
        if (
            overall < threshold
            and faith_result.score >= 0.5
            and rel_result.score >= 0.5
            and not tox_result.is_toxic
        ):
            reasons.append(f"LOW QUALITY SCORE: {overall:.0%} (needs >{threshold:.0%})")
            reasons.append("   The answer is faithful and relevant, but below the quality bar.")

        return "\n\n".join(reasons)

    def score(
        self,
        question: str,   # what the user asked
        context: str,    # what information the LLM was given
        answer: str,     # what the LLM responded
    ) -> EvalResult:
        """
        Evaluate one LLM output completely.
        Runs all three scorers and returns a combined result.
        """
        start_time = time.time()

        # Run all three scorers
        faith_result = self._faithfulness.score(context, answer)
        rel_result   = self._relevance.score(question, answer)
        tox_result   = self._toxicity.score(answer)

        # safety_score: 1.0 = completely safe, 0.0 = completely toxic
        safety_score = 1.0 - tox_result.score

        # Weighted combination
        overall = (
            (faith_result.score * self.fw) +
            (rel_result.score   * self.rw) +
            (safety_score       * self.sw)
        )
        overall = round(max(0.0, min(1.0, overall)), 4)

        evaluation_duration = round(time.time() - start_time, 3)

        result = EvalResult(
            overall=overall,
            faithfulness=faith_result,
            relevance=rel_result,
            toxicity=tox_result,
            weights={
                "faithfulness": self.fw,
                "relevance": self.rw,
                "safety": self.sw,
            },
        )

        # Persistence is opt-in — a score never requires a database.
        if self.save:
            threshold = 0.6
            passed = result.passed(threshold)
            failure_reason = self._build_failure_reason(
                question=question,
                context=context,
                answer=answer,
                faith_result=faith_result,
                rel_result=rel_result,
                tox_result=tox_result,
                overall=overall,
                passed=passed,
                threshold=threshold,
            )

            save_evaluation(
                question=question,
                answer=answer,
                context=context,
                faithfulness=faith_result.score,
                relevance=rel_result.score,
                toxicity=tox_result.score,
                overall_score=overall,
                passed=passed,
                failure_reason=failure_reason,
                model_name="minieval-v1",
                model_temperature=0.7,
                prompt_template="default",
                evaluation_duration=evaluation_duration,
            )

        return result

    def score_batch(
        self,
        items: list[dict],  # list of {question, context, answer}
    ) -> list[EvalResult]:
        """
        Evaluate multiple outputs at once.

        Usage:
            results = ev.score_batch([
                {"question": "...", "context": "...", "answer": "..."},
                {"question": "...", "context": "...", "answer": "..."},
            ])
        """
        return [
            self.score(
                question=item["question"],
                context=item["context"],
                answer=item["answer"],
            )
            for item in items
        ]