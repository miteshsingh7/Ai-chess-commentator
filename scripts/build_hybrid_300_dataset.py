"""Assemble the hybrid 300-position intermediate dataset (natural games + Lichess puzzles)."""

import os
import json
import random
import chess
import pandas as pd
from collections import Counter
from typing import List, Dict, Any

from chess_commentator.analysis.features import extract_board_features
from chess_commentator.analysis.taxonomy import classify_position_taxonomy
from chess_commentator.analysis.models import (
    PositionAnalysis,
    MoveMetadata,
    BoardFeatures,
    TaxonomyResult,
)

PUZZLES_PATH = "data/lichess_tactical_puzzles.json"
NATURAL_GAMES_PATH = "/Users/miteshsingh/Documents/projects/chess_analyzer/data/ThEPheN0MiNaL/moves_categorized.parquet"
OUTPUT_PARQUET = "data/intermediate_300_hybrid.parquet"

NATURAL_TARGETS = {
    "none": 54,
    "hanging_piece": 30,
    "missed_fork": 25,
    "missed_sacrifice": 25,
    "missed_mate": 25,
    "zwischenzug": 15,
    "overloaded_piece": 15,
    "trapped_piece": 10,
    "trapped_bishop": 5,
    "trapped_knight": 5,
    "trapped_rook": 5,
    "mistake_other": 15,
}


def build_hybrid_dataset() -> pd.DataFrame:
    random.seed(42)
    analyses: List[PositionAnalysis] = []

    # 1. Process Lichess Tactical Puzzles (96 positions)
    print("Loading Lichess puzzles...")
    with open(PUZZLES_PATH, "r", encoding="utf-8") as f:
        puzzles = json.load(f)

    for p in puzzles:
        b = chess.Board(p["fen"])
        move = chess.Move.from_uci(p["move_uci"])
        color = b.turn
        color_str = "white" if color == chess.WHITE else "black"

        features = extract_board_features(board=b, color=color, move_number=20)
        tax = classify_position_taxonomy(
            board=b,
            move=move,
            player_color=color,
            eval_before=650,
            eval_after=650,
            cp_loss=0,
            best_move=move,
            played_best=True,
            features=features,
        )

        metadata = MoveMetadata(
            game_id=f"lichess_{p['puzzle_id']}",
            move_number=20,
            player_color=color_str,
            move_san=p["move_san"],
            move_uci=p["move_uci"],
            result="1-0" if color == chess.WHITE else "0-1",
            opening="Tactical Puzzle",
        )

        analysis = PositionAnalysis(
            fen=p["fen"],
            move_uci=p["move_uci"],
            move_san=p["move_san"],
            player_color=color_str,
            move_number=20,
            eval_before=650,
            eval_after=650,
            cp_loss=0,
            best_move=p["move_uci"],
            played_best=True,
            mistake_type="good",
            taxonomy=tax,
            features=features,
            metadata=metadata,
        )
        analyses.append(analysis)

    print(f"Added {len(analyses)} puzzle analyses.")

    # 2. Process Natural Human Games (204 positions)
    print("Loading natural amateur games...")
    df_nat = pd.read_parquet(NATURAL_GAMES_PATH)
    print(f"Loaded {len(df_nat)} moves from natural games.")

    # Reclassify with audited taxonomy
    classified_natural = []
    for r in df_nat.to_dict("records"):
        b = chess.Board(r["fen"])
        move = chess.Move.from_uci(r["move_uci"])
        color = chess.WHITE if r["player_color"] == "white" else chess.BLACK
        best_uci = r.get("best_move")
        best_move = chess.Move.from_uci(best_uci) if best_uci else None
        loss = r.get("cp_loss", 0)
        eval_before = r.get("eval_before")
        eval_after = r.get("eval_after")
        played_best = bool(r.get("played_best", False))

        tax = classify_position_taxonomy(
            board=b,
            move=move,
            player_color=color,
            eval_before=eval_before,
            eval_after=eval_after,
            cp_loss=loss,
            best_move=best_move,
            played_best=played_best,
            mode="deep",
        )
        r["tax_result"] = tax
        r["tactic_type"] = tax.tactic_type
        classified_natural.append(r)

    # Sample into targets
    nat_pools = {k: [] for k in NATURAL_TARGETS}
    for r in classified_natural:
        t = r["tactic_type"]
        if t in ("knight_fork", "pawn_fork", "fork") and r.get("cp_loss", 0) >= 150:
            nat_pools["missed_fork"].append(r)
        elif t == "missed_sacrifice":
            nat_pools["missed_sacrifice"].append(r)
        elif t == "missed_mate":
            nat_pools["missed_mate"].append(r)
        elif t == "zwischenzug":
            nat_pools["zwischenzug"].append(r)
        elif t == "hanging_piece":
            nat_pools["hanging_piece"].append(r)
        elif t == "overloaded_piece":
            nat_pools["overloaded_piece"].append(r)
        elif t == "trapped_bishop":
            nat_pools["trapped_bishop"].append(r)
            nat_pools["trapped_piece"].append(r)
        elif t == "trapped_knight":
            nat_pools["trapped_knight"].append(r)
            nat_pools["trapped_piece"].append(r)
        elif t == "trapped_rook":
            nat_pools["trapped_rook"].append(r)
            nat_pools["trapped_piece"].append(r)
        elif t == "trapped_piece":
            nat_pools["trapped_piece"].append(r)
        elif t == "none" and r.get("played_best", False):
            nat_pools["none"].append(r)
        elif t in ("mistake_other", "blunder_other"):
            nat_pools["mistake_other"].append(r)


    for cat, quota in NATURAL_TARGETS.items():
        pool = nat_pools[cat]
        random.shuffle(pool)
        selected = pool[:quota]
        print(f"  Natural {cat:18}: selected {len(selected)}/{quota} (pool size: {len(pool)})")

        for r in selected:
            b = chess.Board(r["fen"])
            color = chess.WHITE if r["player_color"] == "white" else chess.BLACK
            color_str = "white" if color == chess.WHITE else "black"
            features = extract_board_features(
                board=b,
                color=color,
                move_number=r.get("move_number", 20),
                time_left=r.get("time_left"),
            )
            tax: TaxonomyResult = r["tax_result"]

            meta = MoveMetadata(
                game_id=str(r.get("game_id", "game_unknown")),
                move_number=int(r.get("move_number", 20)),
                player_color=color_str,
                move_san=r["move_san"],
                move_uci=r["move_uci"],
                time_left=r.get("time_left"),
                result=str(r.get("result", "*")),
                eco=str(r.get("eco", "?")),
                opening=str(r.get("opening", "?")),
            )

            analysis = PositionAnalysis(
                fen=r["fen"],
                move_uci=r["move_uci"],
                move_san=r["move_san"],
                player_color=color_str,
                move_number=int(r.get("move_number", 20)),
                eval_before=r.get("eval_before"),
                eval_after=r.get("eval_after"),
                cp_loss=r.get("cp_loss"),
                best_move=r.get("best_move"),
                played_best=bool(r.get("played_best", False)),
                mistake_type=str(r.get("mistake_type", "good" if tax.tactic_type == "none" else "blunder")),
                taxonomy=tax,
                features=features,
                metadata=meta,
            )
            analyses.append(analysis)

    print(f"\nTotal assembled analyses: {len(analyses)}")
    records = [a.to_dict() for a in analyses]
    df = pd.DataFrame(records)

    # Verify deduplication
    print(f"Checking deduplication on (fen, move_uci)...")
    unique_fen_moves = df.drop_duplicates(subset=["fen", "move_uci"])
    print(f"Total rows: {len(df)} | Unique (fen, move_uci): {len(unique_fen_moves)}")

    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Saved dataset to {OUTPUT_PARQUET}")

    print("\nTaxonomy Distribution:")
    print(df["tactic_type"].value_counts())
    return df

if __name__ == "__main__":
    build_hybrid_dataset()
