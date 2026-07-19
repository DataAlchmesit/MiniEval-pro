from __future__ import annotations
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer



@dataclass
class RelevanceResult:
    score: float        # 0.0 to 1.0 — higher means more relevant
    label: str          # "relevant" | "partial" | "irrelevant"
    explanation: str


class RelevanceScorer:
    """
    Checks if the LLM answer actually addresses the question asked.
    Uses sentence embeddings + cosine similarity.
    Model: all-MiniLM-L6-v2 — only 80MB, very fast.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self._model = None  # lazy load again

    def _load_model(self):
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer
        print("[MiniEval] Loading relevance model (~80MB, one-time only)...")
        self._model = SentenceTransformer(self.MODEL_NAME)
        print("[MiniEval] Relevance model ready.")

    def score(
        self,
        question: str,  # the original question
        answer: str,    # the LLM's answer
    ) -> RelevanceResult:
        """
        Score how relevant the answer is to the question.

        Example:
            question = "What is photosynthesis?"
            answer   = "Photosynthesis is how plants make food from sunlight."
            result.score → 0.89  (highly relevant)

            question = "What is photosynthesis?"
            answer   = "The weather in Paris is 22 degrees today."
            result.score → 0.12  (irrelevant)
        """
        self._load_model()

        # encode() converts text into a vector (a list of 384 numbers)
        # These numbers capture the MEANING of the text
        # Similar meanings → similar numbers → similar direction
        embeddings = self._model.encode(
            [question, answer],
            convert_to_tensor=True,  # return as PyTorch tensor
            normalize_embeddings=True,  # normalize so cosine = dot product
        )

        question_vec = embeddings[0]
        answer_vec   = embeddings[1]

        # Cosine similarity: dot product of two normalized vectors
        # Result is between -1 (opposite) and 1 (identical)
        # We use .item() to convert from tensor to plain Python float
        similarity = float((question_vec * answer_vec).sum().item())

        # Shift from [-1, 1] range to [0, 1] range
        # Formula: (x + 1) / 2
        relevance_score = (similarity + 1) / 2

        # Label based on thresholds
        # These thresholds come from empirical testing — adjust as you
        # collect more real data from your users
        if relevance_score >= 0.7:
            label = "relevant"
            explanation = (
                f"The answer directly addresses the question. "
                f"(similarity: {relevance_score:.0%})"
            )
        elif relevance_score >= 0.4:
            label = "partial"
            explanation = (
                f"The answer partially addresses the question "
                f"but may be missing key information. "
                f"(similarity: {relevance_score:.0%})"
            )
        else:
            label = "irrelevant"
            explanation = (
                f"The answer does not address the question. "
                f"(similarity: {relevance_score:.0%})"
            )

        return RelevanceResult(
            score=round(relevance_score, 4),
            label=label,
            explanation=explanation,
        )