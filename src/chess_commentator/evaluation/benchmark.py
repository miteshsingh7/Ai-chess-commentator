"""Comparative benchmark runner for evaluating commentary models."""

from typing import List, Dict, Any, Optional
import pandas as pd
from chess_commentator.evaluation.metrics import (
    compute_text_metrics,
    compute_eval_sentiment_agreement,
    compute_tactic_recall,
    compute_hallucination_rate,
    EvaluationSummary,
)
from chess_commentator.evaluation.judge import LLMJudge, JudgeScore


def run_benchmark(
    test_df: pd.DataFrame,
    candidate_predictions: List[str],
    reference_column: str = "teacher_commentary",
    judge: Optional[LLMJudge] = None,
    num_judge_samples: int = 10,
) -> Dict[str, Any]:
    """Run comprehensive automated metrics and LLM judge on test predictions."""
    references = test_df[reference_column].tolist() if reference_column in test_df else [""] * len(test_df)
    mistake_types = test_df.get("mistake_type", pd.Series(["good"] * len(test_df))).tolist()
    tactic_types = test_df.get("tactic_type", pd.Series(["none"] * len(test_df))).tolist()
    fens = test_df["fen"].tolist()
    moves = test_df["move_uci"].tolist()

    # 1. Automated NLP & Domain metrics
    text_metrics = compute_text_metrics(candidate_predictions, references)
    agreement = compute_eval_sentiment_agreement(candidate_predictions, mistake_types)
    recall = compute_tactic_recall(candidate_predictions, tactic_types)
    hallucination = compute_hallucination_rate(candidate_predictions, fens, moves)

    summary = EvaluationSummary(
        rouge_l=text_metrics["rouge_l"],
        token_f1=text_metrics["token_f1"],
        eval_sentiment_agreement=agreement,
        tactic_recall=recall,
        hallucination_rate=hallucination,
        total_samples=len(candidate_predictions),
    )

    # 2. LLM Judge on sample subset
    judge_instance = judge or LLMJudge(mock_mode=True)
    sample_indices = range(min(num_judge_samples, len(test_df)))
    judge_scores: List[JudgeScore] = []

    for idx in sample_indices:
        score = judge_instance.evaluate(
            fen=fens[idx],
            move=moves[idx],
            eval_summary=f"{mistake_types[idx]} ({test_df.iloc[idx].get('cp_loss', 0)} cp loss)",
            tactical_motif=tactic_types[idx],
            candidate_commentary=candidate_predictions[idx],
            reference_commentary=references[idx],
        )
        judge_scores.append(score)

    mean_judge_overall = (
        round(sum(s.overall for s in judge_scores) / len(judge_scores), 2)
        if judge_scores else 0.0
    )

    result = {
        "summary": summary.to_dict(),
        "mean_judge_score": mean_judge_overall,
        "sample_judge_details": [s.to_dict() for s in judge_scores],
    }
    return result
