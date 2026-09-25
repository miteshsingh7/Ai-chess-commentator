"""Run teacher commentary generation on the 300-position hybrid dataset and audit phrase diversity."""

import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from chess_commentator.teacher.client import GroqTeacherClient
from chess_commentator.teacher.generator import TeacherCommentaryGenerator
from chess_commentator.dataset.filters import PhraseDiversityChecker
from chess_commentator.analysis.models import (
    PositionAnalysis,
    BoardFeatures,
    TaxonomyResult,
    MoveMetadata,
)

INPUT_PARQUET = "data/intermediate_300_hybrid.parquet"
OUTPUT_PARQUET = "data/intermediate_300_with_commentary.parquet"

def safe_optional_int(val):
    if val is None:
        return None
    try:
        import math
        if math.isnan(val):
            return None
        return int(val)
    except (TypeError, ValueError):
        return None


def row_to_analysis(r: dict) -> PositionAnalysis:
    tax = TaxonomyResult(
        mistake_category=r.get("mistake_category", "none"),
        tactic_type=r.get("tactic_type", "none"),
        piece_lost=r.get("piece_lost", "none"),
        mate_missed=int(r.get("mate_missed", 0)),
    )
    features = BoardFeatures(
        phase=r.get("phase", "middlegame"),
        castled=bool(r.get("castled", False)),
        open_files_near_king=int(r.get("open_files_near_king", 0)),
        doubled_pawns=int(r.get("doubled_pawns", 0)),
        isolated_pawns=int(r.get("isolated_pawns", 0)),
        passed_pawns=int(r.get("passed_pawns", 0)),
        mobility=int(r.get("mobility", 20)),
        time_pressure=bool(r.get("time_pressure", False)),
        pawns=int(r.get("pawns", 8)),
        knights=int(r.get("knights", 2)),
        bishops=int(r.get("bishops", 2)),
        rooks=int(r.get("rooks", 2)),
        queens=int(r.get("queens", 1)),
        opp_pawns=int(r.get("opp_pawns", 8)),
        opp_knights=int(r.get("opp_knights", 2)),
        opp_bishops=int(r.get("opp_bishops", 2)),
        opp_rooks=int(r.get("opp_rooks", 2)),
        opp_queens=int(r.get("opp_queens", 1)),
        material_balance=int(r.get("material_balance", 0)),
    )
    meta = MoveMetadata(
        game_id=str(r.get("game_id", "unknown")),
        move_number=int(r.get("move_number", 20)),
        player_color=str(r.get("player_color", "white")),
        move_san=str(r.get("move_san", "")),
        move_uci=str(r.get("move_uci", "")),
        time_left=r.get("time_left"),
        result=str(r.get("result", "*")),
        eco=str(r.get("eco", "?")),
        opening=str(r.get("opening", "?")),
    )
    return PositionAnalysis(
        fen=r["fen"],
        move_uci=r["move_uci"],
        move_san=r["move_san"],
        player_color=r["player_color"],
        move_number=int(r.get("move_number", 20)),
        eval_before=safe_optional_int(r.get("eval_before")),
        eval_after=safe_optional_int(r.get("eval_after")),
        cp_loss=safe_optional_int(r.get("cp_loss")),
        best_move=r.get("best_move"),
        played_best=bool(r.get("played_best", False)),
        mistake_type=str(r.get("mistake_type", "good")),
        taxonomy=tax,
        features=features,
        metadata=meta,
    )

def main():
    if not os.path.exists(INPUT_PARQUET):
        print(f"Error: {INPUT_PARQUET} not found. Run scripts/build_hybrid_300_dataset.py first.")
        sys.exit(1)

    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df)} positions from {INPUT_PARQUET}.")

    # Check for existing progress
    completed_records = {}
    if os.path.exists(OUTPUT_PARQUET):
        try:
            df_existing = pd.read_parquet(OUTPUT_PARQUET)
            for rec in df_existing.to_dict("records"):
                if rec.get("teacher_commentary"):
                    completed_records[(rec["fen"], rec["move_uci"])] = rec
            print(f"Found {len(completed_records)} existing generated positions.")
        except Exception as e:
            print(f"Could not load existing progress: {e}")

    client = GroqTeacherClient()
    generator = TeacherCommentaryGenerator(client=client)

    all_rows = []
    print(f"Generating commentary for {len(df)} positions...")

    for idx, row in enumerate(df.to_dict("records")):
        key = (row["fen"], row["move_uci"])
        if key in completed_records:
            all_rows.append(completed_records[key])
            continue

        analysis = row_to_analysis(row)
        comm, model_used, reason = generator.generate_for_position(analysis)
        rec = dict(row)
        rec["teacher_commentary"] = comm
        rec["teacher_model"] = model_used
        rec["teacher_routing_reason"] = reason
        all_rows.append(rec)
        completed_records[key] = rec

        # Checkpoint every 10 positions
        if len(all_rows) % 10 == 0:
            pd.DataFrame(all_rows).to_parquet(OUTPUT_PARQUET, index=False)
            print(f"  Checkpoint: {len(all_rows)}/{len(df)} saved.")

    df_out = pd.DataFrame(all_rows)
    df_out.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nAll {len(df_out)} positions generated and saved to {OUTPUT_PARQUET}!")

    # Print spend summary
    spend = client.get_spend_summary()
    print("\n--- Groq Spend Summary ---")
    print(f"Total Calls: {spend['total_calls']}")
    for m, c in spend.get("model_breakdown", {}).items():
        print(f"  {m}: {c['calls']} calls, {c['input_tokens']} in, {c['output_tokens']} out, ${c['cost_usd']:.4f}")

    # Stage 3 Phrase Diversity Audit
    print("\n--- Stage 3 Phrase Diversity Audit ---")
    checker = PhraseDiversityChecker(n_gram=3, threshold=0.40)
    audit = checker.audit_dataset(df_out, commentary_col="teacher_commentary", taxonomy_col="tactic_type")

    print(f"Evaluated commentary pairs: {audit.total_pairs_evaluated}")
    print(f"Flagged high-overlap pairs (>40% 3-gram overlap): {audit.flagged_count} ({audit.flagged_rate:.2%})")
    print("\nPer-Category Stats:")
    for cat, stats in audit.category_stats.items():
        print(f"  {cat:20}: max_overlap={stats['max_overlap']:.1%}, avg_overlap={stats['avg_overlap']:.1%}, flagged={stats['flagged_pairs']}/{stats['total_pairs']}")

    if audit.top_overlapping_pairs:
        print("\nTop 5 Overlapping Pairs:")
        for i, p in enumerate(audit.top_overlapping_pairs[:5], 1):
            print(f"  #{i} Category: {p['category']} | Overlap: {p['overlap']:.1%}")
            print(f"     A: {p['text_a'][:100]}...")
            print(f"     B: {p['text_b'][:100]}...")

    # Print final taxonomy distribution
    print("\n--- Final 300 Taxonomy Distribution ---")
    print(df_out["tactic_type"].value_counts())

if __name__ == "__main__":
    main()
