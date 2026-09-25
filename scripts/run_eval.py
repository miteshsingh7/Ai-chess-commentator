"""CLI to evaluate commentary predictions with automated metrics and LLM judge."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import pandas as pd
from chess_commentator.evaluation.benchmark import run_benchmark
from chess_commentator.evaluation.judge import LLMJudge


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate commentary model on test dataset.")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test parquet or jsonl.")
    parser.add_argument("--predictions-col", type=str, default="teacher_commentary", help="Column containing predictions.")
    parser.add_argument("--mock-judge", action="store_true", help="Use mock LLM judge instead of API.")
    args = parser.parse_args()

    if args.test_data.endswith(".parquet"):
        df = pd.read_parquet(args.test_data)
    else:
        df = pd.read_json(args.test_data, lines=True)

    print(f"Loaded {len(df)} evaluation samples.")
    preds = df[args.predictions_col].tolist()

    judge = LLMJudge(mock_mode=args.mock_judge)
    results = run_benchmark(df, preds, reference_column="teacher_commentary", judge=judge)

    print("\n" + "=" * 55)
    print("📊 BENCHMARK EVALUATION REPORT")
    print("=" * 55)
    summary = results["summary"]
    print(f"Total Samples Evaluated:     {summary['total_samples']}")
    print(f"ROUGE-L Score:               {summary['rouge_l']:.4f}")
    print(f"Token F1 Score:              {summary['token_f1']:.4f}")
    print(f"Eval-Sentiment Agreement:   {summary['eval_sentiment_agreement'] * 100:.1f}%")
    print(f"Tactic Recall Rate:          {summary['tactic_recall'] * 100:.1f}%")
    print(f"Hallucination Rate:          {summary['hallucination_rate'] * 100:.1f}%")
    print(f"Mean LLM Judge Score (1-5):  {results['mean_judge_score']:.2f} / 5.0")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
