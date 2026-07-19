from __future__ import annotations
from dataclasses import dataclass
import torch

from transformers import pipeline


# dataclass = a clean container for results
# Instead of returning a messy dict, we return a proper object
# with named fields. Clean API design.
@dataclass
class FaithfulnessResult:
    score: float        # 0.0 to 1.0 — higher means more faithful
    label: str          # "faithful" | "neutral" | "contradicts"
    confidence: float   # how sure the model is
    explanation: str    # human readable reason

    # Raw NLI probabilities, exposed so callers can set their own
    # thresholds rather than inheriting ours.
    entailment: float = 0.0
    contradiction: float = 0.0
    neutral: float = 0.0


class FaithfulnessScorer:
    """
    Checks if an LLM answer is faithful to its source context.
    Uses NLI model — no OpenAI key, no API cost.
    """

    # We use DeBERTa-v3-small — best accuracy at this size.
    # 180MB download, 40ms per eval on CPU.
    MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

    def __init__(self, quiet: bool = False):
        # IMPORTANT: we do NOT load the model here.
        # If we loaded in __init__, every import of this file
        # would trigger a 180MB download. That would feel broken.
        # We load lazily — only when .score() is first called.
        self._model = None
        self.quiet = quiet

    def _load_model(self):
        """Load the model only when first needed."""
        if self._model is not None:
            return  # already loaded, skip

        from transformers import pipeline
        if not self.quiet:
            print("[MiniEval] Loading faithfulness model (~180MB, one-time only)...")

        self._model = pipeline(
            task="text-classification",
            model=self.MODEL_NAME,
            # Use GPU if available, otherwise CPU
            device=0 if torch.cuda.is_available() else -1,
            # top_k=None means: return ALL label scores, not just the top 1
            top_k=None,
        )
        if not self.quiet:
            print("[MiniEval] Faithfulness model ready.")

    def score(
        self,
        context: str,   # the source text (your RAG chunks, documents, etc.)
        answer: str,    # the LLM output you want to evaluate
        threshold: float = 0.5,  # above this = faithful
    ) -> FaithfulnessResult:
        """
        Score whether the answer is faithful to the context.

        Example:
            context = "The Eiffel Tower is 330m tall and is in Paris."
            answer  = "The Eiffel Tower is located in Paris."
            result.score  → 0.91
            result.label  → "faithful"
        """
        # Load model on first call
        self._load_model()

        # NLI input format: "premise [SEP] hypothesis"
        # premise   = context (what we know is true)
        # hypothesis = answer (what we want to verify)
        # Question we are asking the model:
        # "Given this context, does this answer follow?"
        nli_input = f"{context} [SEP] {answer}"

        # Run the model — returns list of {label, score} dicts
        raw_output = self._model(nli_input)[0]

        # Convert to a simple dict: {"entailment": 0.87, ...}
        scores = {
            item["label"].lower(): item["score"]
            for item in raw_output
        }

        entailment    = scores.get("entailment", 0.0)
        contradiction = scores.get("contradiction", 0.0)
        neutral       = scores.get("neutral", 0.0)

        # Calculate faithfulness score.
        # Core idea: entailment is good, contradiction is very bad.
        # We penalize contradiction at 0.5x weight.
        # Why 0.5? A small contradiction should not completely
        # destroy the score — but it should hurt it meaningfully.
        faith_score = entailment - (contradiction * 0.5)

        # Clamp to 0.0–1.0 range — math can push outside bounds
        faith_score = max(0.0, min(1.0, faith_score))

        # Decide the human-readable label
        if contradiction > 0.5:
            label = "contradicts"
            explanation = (
                f"The answer contradicts the source context. "
                f"The LLM likely hallucinated. "
                f"(contradiction confidence: {contradiction:.0%})"
            )
        elif entailment >= threshold:
            label = "faithful"
            explanation = (
                f"The answer is supported by the source context. "
                f"(entailment confidence: {entailment:.0%})"
            )
        else:
            label = "neutral"
            explanation = (
                f"The answer is not clearly supported or contradicted. "
                f"It may contain information not present in the context. "
                f"(neutral confidence: {neutral:.0%})"
            )

        return FaithfulnessResult(
            score=round(faith_score, 4),
            label=label,
            confidence=round(max(entailment, contradiction, neutral), 4),
            explanation=explanation,
            entailment=round(entailment, 4),
            contradiction=round(contradiction, 4),
            neutral=round(neutral, 4),
        )