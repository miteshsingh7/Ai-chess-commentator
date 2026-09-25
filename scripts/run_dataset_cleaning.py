"""CLI to clean, filter, balance, and format commentary datasets for training."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import pandas as pd
from chess_commentator.dataset.filters import (
    validate_eval_sign_consistency,
    validate_filler_and_length,
)
from chess_commentator.dataset.checker import validate_chess_grounding
from chess_commentator.dataset.balancer import balance_and_split_dataset
from chess_commentator.dataset.formatter import export_sft_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and curate synthetic commentary dataset.")
    parser.add_argument("--input", type=str, required=True, help="Input Parquet with commentary.")
    parser.add_argument("--output-dir", type=str, default="data/dataset", help="Output directory for JSONL splits.")
    parser.add_argument("--max-per-cat", type=int, default=500, help="Max items per tactical category.")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} candidate pairs.")

    passed_rows = []
    rejected_stats = {}

    for row in df.to_dict("records"):
        comm = row.get("teacher_commentary", "")
        loss = row.get("cp_loss")
        best = bool(row.get("played_best", False))
        mtype = row.get("mistake_type", "good")
        fen = row.get("fen", "")
        move = row.get("move_uci", "")
        best_uci = row.get("best_move")

        # Filter 1: Filler and length
        f1 = validate_filler_and_length(comm)
        if not f1.passed:
            rejected_stats["filler_length"] = rejected_stats.get("filler_length", 0) + 1
            continue

        # Filter 2: Eval sign consistency
        f2 = validate_eval_sign_consistency(comm, loss, best, mtype)
        if not f2.passed:
            rejected_stats["eval_sign"] = rejected_stats.get("eval_sign", 0) + 1
            continue

        # Filter 3: Chess grounding
        f3 = validate_chess_grounding(fen, move, comm, best_uci)
        if not f3.passed:
            rejected_stats["hallucination"] = rejected_stats.get("hallucination", 0) + 1
            continue

        passed_rows.append(row)

    print(f"Quality filtering complete:")
    print(f"  Retained: {len(passed_rows)} / {len(df)}")
    print(f"  Rejections breakdown: {rejected_stats}")

    clean_df = pd.DataFrame(passed_rows)
    train_df, val_df, test_df = balance_and_split_dataset(
        clean_df,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        max_per_category=args.max_per_cat,
    )

    print(f"Splits created: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    os.makedirs(args.output_dir, exist_ok=True)
    export_sft_dataset(train_df, os.path.join(args.output_dir, "train.jsonl"))
    export_sft_dataset(val_df, os.path.join(args.output_dir, "val.jsonl"))
    export_sft_dataset(test_df, os.path.join(args.output_dir, "test.jsonl"))
    print(f"Exported ChatML JSONL datasets to: {args.output_dir}")


if __name__ == "__main__":
    main()
