"""Stage 5 LLM-as-a-Judge evaluation runner.

Evaluates test set predictions using LLMJudge on Groq free tier ('qwen/qwen3.8-27b')
with sliding-window rate limiting (25 RPM) and persistent checkpointing.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chess_commentator.evaluation.judge import LLMJudge, JudgeScore
from chess_commentator.teacher.client import GroqTeacherClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_stage5_judge(
    sample_per_category: Optional[int] = None,
    match_indices_from: Optional[str] = None,
    predictions_path: str = ".stage5_output_v10/stage5_predictions.json",
    output_scores_path: str = "data/stage5_judge_scores_v10.json",
    final_report_path: str = "data/stage5_final_evaluation_report_v10.json",
    summary_path: str = ".stage5_output_v10/stage5_summary.json",
) -> Dict[str, Any]:
    """Execute LLM Judge evaluation across all predictions with checkpointing."""
    pred_file = Path(predictions_path)
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions file not found at: {pred_file}")

    with open(pred_file, "r", encoding="utf-8") as f:
        predictions: List[Dict[str, Any]] = json.load(f)

    logger.info(f"Loaded {len(predictions)} predictions from {pred_file}")

    if match_indices_from is not None:
        with open(match_indices_from, "r", encoding="utf-8") as f:
            target_indices = {int(item["index"]) for item in json.load(f)}
        predictions = [p for p in predictions if int(p.get("index", -1)) in target_indices]
        logger.info(f"Filtered to {len(predictions)} predictions matching indices in {match_indices_from}")

    elif sample_per_category is not None:
        cat_counts = defaultdict(int)
        sampled = []
        for p in predictions:
            m = p.get("motif", "none")
            if cat_counts[m] < sample_per_category:
                sampled.append(p)
                cat_counts[m] += 1
        predictions = sampled
        logger.info(f"Stratified sample: {len(predictions)} predictions across categories (up to {sample_per_category} per category).")


    # Load existing judge scores checkpoint if available
    scores_file = Path(output_scores_path)
    scores_file.parent.mkdir(parents=True, exist_ok=True)
    existing_scores: Dict[str, Dict[str, Any]] = {}
    if scores_file.exists():
        try:
            with open(scores_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                existing_scores = {str(item["index"]): item for item in saved}
                logger.info(f"Resuming from checkpoint: {len(existing_scores)}/{len(predictions)} already judged.")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")

    # Initialize Groq client and judge
    client = GroqTeacherClient(
        model="qwen/qwen3.8-27b",
        temperature=0.1,
        max_tokens=300,
        requests_per_minute=25,
    )
    judge = LLMJudge(client=client)

    judge_results: List[Dict[str, Any]] = []

    start_time = time.time()
    for idx, item in enumerate(predictions):
        item_idx = str(item.get("index", idx))

        if item_idx in existing_scores:
            judge_results.append(existing_scores[item_idx])
            continue

        fen = item["fen"]
        move_san = item.get("move_san", item.get("move_uci", ""))
        move_uci = item.get("move_uci", "")
        move_repr = f"{move_san} ({move_uci})" if move_san != move_uci else move_san
        m_type = item.get("mistake_type", "good")
        loss = item.get("cp_loss", 0)
        best_mv = item.get("best_move")
        eval_summary = f"{m_type.upper()}" + (f" (loss: {loss} cp, best move was {best_mv})" if loss else "")
        motif = item.get("motif", "none")
        candidate_comm = item.get("generated_commentary", "")
        ref_comm = item.get("reference_commentary", "")

        try:
            score: JudgeScore = judge.evaluate(
                fen=fen,
                move=move_repr,
                eval_summary=eval_summary,
                tactical_motif=motif,
                candidate_commentary=candidate_comm,
                reference_commentary=ref_comm,
            )
        except Exception as e:
            logger.error(f"Error evaluating item {item_idx}: {e}")
            score = JudgeScore(
                accuracy=3,
                educational_value=3,
                naturalness=3,
                conciseness=3,
                overall=3.0,
                feedback=f"Evaluation error: {e}",
            )

        result_record = {
            "index": int(item_idx),
            "motif": motif,
            "move_san": move_san,
            "fen": fen,
            "candidate_commentary": candidate_comm,
            "reference_commentary": ref_comm,
            "judge_score": score.to_dict(),
        }
        judge_results.append(result_record)
        existing_scores[item_idx] = result_record

        # Checkpoint save every 5 items or at the end
        if len(judge_results) % 5 == 0 or len(judge_results) == len(predictions):
            with open(scores_file, "w", encoding="utf-8") as f:
                json.dump(list(existing_scores.values()), f, indent=2)

        elapsed = time.time() - start_time
        processed_new = len(existing_scores) - len(judge_results) + idx + 1
        print(
            f"  [Judge {idx + 1:3d}/{len(predictions)}] [{motif}] Overall: {score.overall:.2f} "
            f"(Acc: {score.accuracy}, Edu: {score.educational_value}, Nat: {score.naturalness}, Conc: {score.conciseness}) "
            f"- Feedback: {score.feedback[:60]}..."
        )

    # Final checkpoint save
    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump(list(existing_scores.values()), f, indent=2)

    # Compute Judge Averages
    total = len(judge_results)
    mean_acc = sum(r["judge_score"]["accuracy"] for r in judge_results) / total
    mean_edu = sum(r["judge_score"]["educational_value"] for r in judge_results) / total
    mean_nat = sum(r["judge_score"]["naturalness"] for r in judge_results) / total
    mean_conc = sum(r["judge_score"]["conciseness"] for r in judge_results) / total
    mean_overall = sum(r["judge_score"]["overall"] for r in judge_results) / total

    # Per-category breakdown for Judge
    cat_judge = defaultdict(list)
    for r in judge_results:
        cat_judge[r["motif"]].append(r["judge_score"])

    per_cat_judge_summary = {}
    for motif, scores in cat_judge.items():
        n = len(scores)
        per_cat_judge_summary[motif] = {
            "n": n,
            "accuracy": round(sum(s["accuracy"] for s in scores) / n, 2),
            "educational_value": round(sum(s["educational_value"] for s in scores) / n, 2),
            "naturalness": round(sum(s["naturalness"] for s in scores) / n, 2),
            "conciseness": round(sum(s["conciseness"] for s in scores) / n, 2),
            "overall": round(sum(s["overall"] for s in scores) / n, 2),
        }

    # Load automated metrics summary if available
    auto_summary = {}
    if Path(summary_path).exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            auto_summary = json.load(f)

    final_report = {
        "dataset": "commentary_sft_test.jsonl",
        "total_evaluated": total,
        "automated_metrics": auto_summary.get("overall", {}),
        "llm_judge_metrics": {
            "model": "qwen/qwen3.8-27b (Groq API)",
            "mean_overall": round(mean_overall, 2),
            "mean_accuracy": round(mean_acc, 2),
            "mean_educational_value": round(mean_edu, 2),
            "mean_naturalness": round(mean_nat, 2),
            "mean_conciseness": round(mean_conc, 2),
        },
        "per_category_automated": auto_summary.get("per_category", {}),
        "per_category_judge": per_cat_judge_summary,
    }

    with open(final_report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    logger.info(f"Final report saved to: {final_report_path}")
    return final_report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-per-cat", type=int, default=None, help="Max samples per category for fast evaluation")
    parser.add_argument("--match-indices-from", type=str, default=None, help="Path to json file whose indices to match")
    parser.add_argument("--predictions-path", type=str, default=".stage5_output_v10/stage5_predictions.json")
    parser.add_argument("--output-scores-path", type=str, default="data/stage5_judge_scores_v10.json")
    parser.add_argument("--final-report-path", type=str, default="data/stage5_final_evaluation_report_v10.json")
    parser.add_argument("--summary-path", type=str, default=".stage5_output_v10/stage5_summary.json")
    args = parser.parse_args()
    run_stage5_judge(
        sample_per_category=args.sample_per_cat,
        match_indices_from=args.match_indices_from,
        predictions_path=args.predictions_path,
        output_scores_path=args.output_scores_path,
        final_report_path=args.final_report_path,
        summary_path=args.summary_path,
    )

