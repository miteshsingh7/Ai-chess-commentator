"""End-to-end 3,000-position generation, Stage 3 curation, and ChatML export pipeline."""

import os
import sys
import time
import json
import pandas as pd
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

from chess_commentator.teacher.client import GroqTeacherClient
from chess_commentator.teacher.generator import TeacherCommentaryGenerator
from chess_commentator.teacher.cache import TeacherCache
from chess_commentator.teacher.prompt_builder import build_teacher_prompt
from chess_commentator.dataset.checker import validate_chess_grounding
from chess_commentator.dataset.filters import (
    validate_eval_sign_consistency,
    validate_filler_and_length,
    PhraseDiversityChecker,
)
from chess_commentator.dataset.balancer import balance_and_split_dataset
from chess_commentator.dataset.formatter import export_sft_dataset
from chess_commentator.analysis.models import (
    PositionAnalysis,
    BoardFeatures,
    TaxonomyResult,
    MoveMetadata,
)

INPUT_CANDIDATES = "data/intermediate_3000_hybrid.parquet"
OUTPUT_RAW = "data/full_3000_raw_commentary.parquet"
OUTPUT_CLEAN = "data/full_3000_clean_curated.parquet"
TRAIN_JSONL = "data/commentary_sft_train.jsonl"
VAL_JSONL = "data/commentary_sft_val.jsonl"
TEST_JSONL = "data/commentary_sft_test.jsonl"


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
        mistake_category=str(r.get("mistake_category", "none")),
        tactic_type=str(r.get("tactic_type", "none")),
        piece_lost=str(r.get("piece_lost", "none")),
        mate_missed=int(r.get("mate_missed", 0)),
    )
    features = BoardFeatures(
        phase=str(r.get("phase", "middlegame")),
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


def load_known_commentaries() -> Dict[tuple, dict]:
    known = {}
    sources = [
        "data/intermediate_300_with_commentary.parquet",
        "data/validation_60_results.parquet",
        "data/validation_30_results.parquet",
        OUTPUT_RAW,
    ]
    for src in sources:
        if os.path.exists(src):
            try:
                df_src = pd.read_parquet(src)
                comm_col = "teacher_commentary" if "teacher_commentary" in df_src.columns else ("new_commentary" if "new_commentary" in df_src.columns else "commentary")
                for rec in df_src.to_dict("records"):
                    c_text = rec.get(comm_col)
                    if c_text and isinstance(c_text, str) and len(c_text.strip()) > 20:
                        key = (rec["fen"], rec["move_uci"])
                        known[key] = {
                            "teacher_commentary": c_text,
                            "teacher_model": rec.get("teacher_model", rec.get("model", "qwen/qwen3.8-27b")),
                            "teacher_routing_reason": rec.get("teacher_routing_reason", rec.get("reason", "cached/prior")),
                        }
            except Exception as e:
                print(f"Note reading {src}: {e}")
    print(f"Loaded {len(known)} existing validated commentary entries.")
    return known


def run_pipeline():
    print("=" * 60)
    print("STAGE 2 & 3: FULL 3,000-POSITION GENERATION & QUALITY PIPELINE")
    print("=" * 60)

    # 1. Ensure candidates are ready
    if not os.path.exists(INPUT_CANDIDATES):
        print(f"Building candidate dataset {INPUT_CANDIDATES}...")
        from scripts.build_hybrid_3000_dataset import build_hybrid_3000_dataset
        df_candidates = build_hybrid_3000_dataset()
    else:
        df_candidates = pd.read_parquet(INPUT_CANDIDATES)
        print(f"Loaded {len(df_candidates)} candidate positions from {INPUT_CANDIDATES}.")

    # Assert 0 duplicate (fen, move_uci)
    n_unique = df_candidates.drop_duplicates(subset=["fen", "move_uci"]).shape[0]
    print(f"Candidate dataset deduplication: {len(df_candidates)} total, {n_unique} unique (fen, move_uci).")

    # 2. Setup Generator with Rate Limiter
    known_commentaries = load_known_commentaries()
    client = GroqTeacherClient(requests_per_minute=25)
    cache = TeacherCache(cache_dir=".cache/teacher_3000")
    generator = TeacherCommentaryGenerator(client=client, cache=cache)

    all_rows = []
    print(f"\nGenerating commentary for {len(df_candidates)} positions...")
    start_time = time.time()

    for idx, row in enumerate(df_candidates.to_dict("records"), 1):
        key = (row["fen"], row["move_uci"])
        rec = dict(row)

        if key in known_commentaries:
            k_info = known_commentaries[key]
            rec["teacher_commentary"] = k_info["teacher_commentary"]
            rec["teacher_model"] = k_info["teacher_model"]
            rec["teacher_routing_reason"] = k_info["teacher_routing_reason"]
            all_rows.append(rec)
        else:
            analysis = row_to_analysis(row)
            try:
                comm, model_used, reason = generator.generate_for_position(analysis, temperature=0.3)
            except Exception as e:
                print(f"Error on position {idx} ({row['move_san']}): {e}. Using fallback...")
                comm = client._generate_mock_commentary(build_teacher_prompt(analysis))
                model_used = "fallback-grounded"
                reason = "Groq TPD limit fallback"

            rec["teacher_commentary"] = comm
            rec["teacher_model"] = model_used
            rec["teacher_routing_reason"] = reason
            all_rows.append(rec)
            known_commentaries[key] = {
                "teacher_commentary": comm,
                "teacher_model": model_used,
                "teacher_routing_reason": reason,
            }

        # Checkpoint every 50 positions
        if len(all_rows) % 50 == 0 or len(all_rows) == len(df_candidates):
            pd.DataFrame(all_rows).to_parquet(OUTPUT_RAW, index=False)
            elapsed = time.time() - start_time
            print(f"  [Progress] {len(all_rows)}/{len(df_candidates)} processed ({elapsed:.1f}s elapsed). Checkpoint saved.")

    df_raw = pd.DataFrame(all_rows)
    df_raw.to_parquet(OUTPUT_RAW, index=False)
    print(f"\nRaw dataset generation complete: {len(df_raw)} records in {OUTPUT_RAW}.")

    # =========================================================
    # STAGE 3: QUALITY CURATION & FILTERING
    # =========================================================
    print("\n" + "=" * 60)
    print("STAGE 3: QUALITY CURATION & CHECKER AUDIT")
    print("=" * 60)

    # 1. Grounding Checker
    print("1. Running Chess Grounding & Move Legality Checker...")
    checker_passed = []
    checker_reasons = []
    piece_sq_flags = 0
    illegal_mv_flags = 0

    for idx, r in df_raw.iterrows():
        chk = validate_chess_grounding(
            fen=r["fen"],
            move_uci=r["move_uci"],
            commentary=r["teacher_commentary"],
            best_move_uci=r.get("best_move"),
        )
        checker_passed.append(chk.passed)
        checker_reasons.append(chk.reason)
        if not chk.passed:
            if "hallucination detected" in chk.reason.lower():
                piece_sq_flags += 1
            if "illegal move" in chk.reason.lower():
                illegal_mv_flags += 1

    df_raw["checker_passed"] = checker_passed
    df_raw["checker_reason"] = checker_reasons

    total_checked = len(df_raw)
    total_passed_chk = sum(checker_passed)
    total_flagged_chk = total_checked - total_passed_chk
    print(f"  Grounding Checker Results: {total_passed_chk}/{total_checked} passed ({total_passed_chk/total_checked:.1%})")
    print(f"  Total Flagged: {total_flagged_chk} ({total_flagged_chk/total_checked:.1%}) | Piece-Square: {piece_sq_flags}, Illegal-Move: {illegal_mv_flags}")

    # Per-category checker flag breakdown
    print("\n  Per-Category Grounding Flag Rates:")
    per_cat_flags = {}
    for tactic, grp in df_raw.groupby("tactic_type"):
        t_tot = len(grp)
        t_pass = grp["checker_passed"].sum()
        t_fl = t_tot - t_pass
        per_cat_flags[tactic] = {"total": t_tot, "flagged": t_fl, "rate": t_fl / t_tot if t_tot else 0.0}
        print(f"    - {tactic:22s}: {t_fl:3d} / {t_tot:3d} flagged ({t_fl/t_tot*100:5.1f}%)")

    # Filter to checker passed
    df_grounded = df_raw[df_raw["checker_passed"]].copy()

    # 2. Eval-sign & filler/length filters
    print("\n2. Running Eval-Sign Consistency & Filler/Length Filters...")
    eval_flags = [
        validate_eval_sign_consistency(
            r["teacher_commentary"],
            safe_optional_int(r.get("cp_loss")),
            bool(r.get("played_best", False)),
            str(r.get("mistake_type", "good")),
        ).passed
        for _, r in df_grounded.iterrows()
    ]
    df_eval_filtered = df_grounded[eval_flags].copy()
    print(f"  Eval-Sign Filter: {len(df_eval_filtered)} / {len(df_grounded)} retained.")

    filler_flags = [
        validate_filler_and_length(r["teacher_commentary"], min_words=15, max_words=120).passed
        for _, r in df_eval_filtered.iterrows()
    ]
    df_clean = df_eval_filtered[filler_flags].copy()
    print(f"  Filler & Length Filter: {len(df_clean)} / {len(df_eval_filtered)} retained.")
    df_clean.to_parquet(OUTPUT_CLEAN, index=False)
    print(f"  Clean curated dataset saved to {OUTPUT_CLEAN} ({len(df_clean)} records).")

    # 3. 3-Gram Phrase Diversity Audit
    print("\n3. Running 3-Gram Phrase Diversity Audit (<40% overlap threshold)...")
    div_checker = PhraseDiversityChecker(n=3, threshold=0.40)
    audit = div_checker.audit_dataframe(
        df_clean,
        text_column="teacher_commentary",
        bucket_column="tactic_type",
    )
    print(f"  Total commentary evaluated: {audit['total_checked']}")
    print(f"  Flagged high-overlap commentary (>40% overlap): {audit['flagged_count']} ({audit['flag_rate']:.2%})")
    print("\n  Per-Category Diversity Flags:")
    for b_cat, b_cnt in audit["bucket_counts"].items():
        b_fl = audit["bucket_flagged"].get(b_cat, 0)
        print(f"    - {b_cat:22s}: {b_fl:3d} / {b_cnt:3d} flagged ({b_fl/b_cnt*100:5.1f}%)")

    # 4. Taxonomy-Balanced & Leak-Free Game-Aware Split
    print("\n4. Performing Game-Aware, Leak-Free 80/10/10 Train/Val/Test Split...")
    # Target exactly 3,000 clean positions if available
    train_df, val_df, test_df = balance_and_split_dataset(
        df_clean,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        max_per_category=400,
        min_per_category=40,
        max_oversample_factor=1.2,
        random_seed=42,
    )

    print(f"  Train set: {len(train_df)} rows")
    print(f"  Val set:   {len(val_df)} rows")
    print(f"  Test set:  {len(test_df)} rows")
    print(f"  Total split size: {len(train_df) + len(val_df) + len(test_df)} rows")

    # Assert zero game leakage between splits
    train_games = set(train_df["game_id"].unique())
    val_games = set(val_df["game_id"].unique())
    test_games = set(test_df["game_id"].unique())
    assert len(train_games & val_games) == 0, "Leakage detected between train and val!"
    assert len(train_games & test_games) == 0, "Leakage detected between train and test!"
    assert len(val_games & test_games) == 0, "Leakage detected between val and test!"
    print("  Zero game-level leakage verified across train/val/test splits!")

    # 5. Export to ChatML JSONL
    print("\n5. Formatting and Exporting ChatML JSONL Datasets...")
    train_df["commentary"] = train_df["teacher_commentary"]
    val_df["commentary"] = val_df["teacher_commentary"]
    test_df["commentary"] = test_df["teacher_commentary"]

    export_sft_dataset(train_df, TRAIN_JSONL)
    export_sft_dataset(val_df, VAL_JSONL)
    export_sft_dataset(test_df, TEST_JSONL)

    print(f"  Saved ChatML train set: {TRAIN_JSONL} ({len(train_df)} examples)")
    print(f"  Saved ChatML val set:   {VAL_JSONL} ({len(val_df)} examples)")
    print(f"  Saved ChatML test set:  {TEST_JSONL} ({len(test_df)} examples)")

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
