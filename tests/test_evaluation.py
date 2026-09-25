"""Unit tests for Stage 5: Evaluation metrics, LLM Judge, and Benchmark runner."""

import pandas as pd
from chess_commentator.evaluation.metrics import (
    compute_text_metrics,
    compute_eval_sentiment_agreement,
    compute_tactic_recall,
    compute_hallucination_rate,
)
from chess_commentator.evaluation.judge import LLMJudge
from chess_commentator.evaluation.benchmark import run_benchmark


def test_text_metrics():
    preds = ["White develops the knight to f3 controlling the center."]
    refs = ["White plays the knight to f3 to control the center squares."]
    metrics = compute_text_metrics(preds, refs)
    assert metrics["rouge_l"] > 0.5
    assert metrics["token_f1"] > 0.5


def test_eval_sentiment_agreement():
    preds = [
        "A shocking blunder that loses the queen to a knight fork.",
        "Excellent move developing the bishop actively.",
    ]
    mistake_types = ["blunder", "good"]
    agreement = compute_eval_sentiment_agreement(preds, mistake_types)
    assert agreement == 1.0


def test_tactic_recall():
    preds = [
        "White unleashes a deadly knight fork attacking both king and rook.",
        "A quiet developing move.",
    ]
    tactics = ["knight_fork", "none"]
    recall = compute_tactic_recall(preds, tactics)
    assert recall == 1.0


def test_hallucination_rate():
    fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
    moves = ["e2e4"]
    # Bad prediction mentioning non-existent bishop on g7
    preds_bad = ["White coordinates with the bishop on g7 to attack."]
    rate_bad = compute_hallucination_rate(preds_bad, fens, moves)
    assert rate_bad == 1.0

    # Good prediction mentioning pawn on e4
    preds_good = ["White advances the pawn to e4, establishing early central control."]
    rate_good = compute_hallucination_rate(preds_good, fens, moves)
    assert rate_good == 0.0


def test_llm_judge_and_benchmark():
    judge = LLMJudge(mock_mode=True)
    score = judge.evaluate(
        fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        move="e7e5",
        eval_summary="good (0 cp loss)",
        tactical_motif="none",
        candidate_commentary="Black stakes an immediate claim in the center, mirroring White's space advantage.",
    )
    assert score.accuracy == 5
    assert score.overall >= 4.0

    # Benchmark run
    df = pd.DataFrame([{
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "move_uci": "e2e4",
        "mistake_type": "good",
        "tactic_type": "none",
        "teacher_commentary": "White establishes a pawn in the center.",
    }])
    preds = ["White advances the pawn to e4 claiming space."]
    benchmark_res = run_benchmark(df, preds, judge=judge, num_judge_samples=1)

    assert "summary" in benchmark_res
    assert benchmark_res["summary"]["eval_sentiment_agreement"] == 1.0
    assert benchmark_res["mean_judge_score"] > 0
