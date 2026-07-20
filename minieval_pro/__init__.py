# --- minieval_pro/web/__init__.py --- 

"""MiniEval - A minimal evaluation framework for LLM outputs."""

# Import from the scorers folder
from .scorers.faithfulness import FaithfulnessScorer, FaithfulnessResult
from .scorers.relevance import RelevanceScorer, RelevanceResult
from .scorers.toxicity import ToxicityScorer, ToxicityResult

# Import from evaluator
from .evaluator import Evaluator, EvalResult

__version__ = "1.1.0"
__all__ = [
    # Main API
    "Evaluator",
    "EvalResult",
    # Individual scorers
    "FaithfulnessScorer",
    "FaithfulnessResult",
    "RelevanceScorer", 
    "RelevanceResult",
    "ToxicityScorer",
    "ToxicityResult",
]