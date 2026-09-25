"""Evaluation module for automated metrics and LLM-as-a-judge scoring."""

from chess_commentator.evaluation.metrics import (
    compute_text_metrics,
    compute_eval_sentiment_agreement,
    compute_tactic_recall,
    compute_hallucination_rate,
    EvaluationSummary,
)
from chess_commentator.evaluation.judge import (
    LLMJudge,
    JudgeScore,
    JUDGE_RUBRIC_PROMPT,
)
from chess_commentator.evaluation.benchmark import run_benchmark

__all__ = [
    "compute_text_metrics",
    "compute_eval_sentiment_agreement",
    "compute_tactic_recall",
    "compute_hallucination_rate",
    "EvaluationSummary",
    "LLMJudge",
    "JudgeScore",
    "JUDGE_RUBRIC_PROMPT",
    "run_benchmark",
]
