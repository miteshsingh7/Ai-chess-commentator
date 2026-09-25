"""Assemble the hybrid 3,000+ candidate dataset for full-scale generation."""

import os
import json
import random
import chess
import pandas as pd
from typing import List, Dict, Any

from chess_commentator.analysis.features import extract_board_features
from chess_commentator.analysis.taxonomy import classify_position_taxonomy
from chess_commentator.analysis.models import (
    PositionAnalysis,
    MoveMetadata,
    BoardFeatures,
    TaxonomyResult,
)

PUZZLES_PATH = "data/lichess_tactical_puzzles_3000.json"
NATURAL_GAMES_PATH = "/Users/miteshsingh/Documents/projects/chess_analyzer/data/ThEPheN0MiNaL/moves_categorized.parquet"
OUTPUT_PARQUET = "data/intermediate_3000_hybrid.parquet"

def build_hybrid_3000_dataset(random_seed: int = 42) -> pd.DataFrame:
    random.seed(random_seed)
    analyses: List[PositionAnalysis] = []
    seen_fen_moves = set()

    # 1. Process Lichess Tactical Puzzles (Target ~1,100 positions)
    print("Loading Lichess puzzles for 3,000 run...")
    puzzles_file = PUZZLES_PATH if os.path.exists(PUZZLES_PATH) else "data/lichess_tactical_puzzles.json"
    with open(puzzles_file, "r", encoding="utf-8") as f:
        puzzles = json.load(f)

    print(f"Loaded {len(puzzles)} puzzles from {puzzles_file}.")
    for p in puzzles:
        key = (p["fen"], p["move_uci"])
        if key in seen_fen_moves:
            continue
        seen_fen_moves.add(key)

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
            game_id=f"lichess_{p.get('puzzle_id', 'puz')}",
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

    # 2. Process Natural Human Games (11,740 available pool)
    print("Loading natural amateur games...")
    df_nat = pd.read_parquet(NATURAL_GAMES_PATH)
    print(f"Loaded {len(df_nat)} moves from natural games.")

    # Reclassify with audited deep taxonomy
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

    # Group into natural pools
    nat_targets = {
        "hanging_piece": 181,
        "overloaded_piece": 110,
        "missed_sacrifice": 51,
        "missed_mate": 42,
        "zwischenzug": 17,
        "missed_fork": 165,
        "trapped_piece": 68,
        "pin": 160,
        "skewer": 133,
        "discovered_attack": 81,
        "middlegame_calculation": 200,
        "endgame_calculation": 54,
        "rook_endgame": 58,
        "mistake_other": 250,
        "none": 1000,
    }

    pools = {k: [] for k in nat_targets}
    for r in classified_natural:
        t = r["tactic_type"]
        if t in ("knight_fork", "pawn_fork", "fork") and r.get("cp_loss", 0) >= 150:
            pools["missed_fork"].append(r)
        elif t == "missed_sacrifice":
            pools["missed_sacrifice"].append(r)
        elif t == "missed_mate":
            pools["missed_mate"].append(r)
        elif t == "zwischenzug":
            pools["zwischenzug"].append(r)
        elif t == "hanging_piece":
            pools["hanging_piece"].append(r)
        elif t == "overloaded_piece":
            pools["overloaded_piece"].append(r)
        elif "trapped" in t:
            pools["trapped_piece"].append(r)
        elif t == "pin":
            pools["pin"].append(r)
        elif t == "skewer":
            pools["skewer"].append(r)
        elif t == "discovered_attack":
            pools["discovered_attack"].append(r)
        elif t == "none" and r.get("played_best", False):
            pools["none"].append(r)
        elif t in ("mistake_other", "blunder_other"):
            pools["mistake_other"].append(r)
        elif t == "middlegame_calculation":
            pools["middlegame_calculation"].append(r)
        elif t == "endgame_calculation":
            pools["endgame_calculation"].append(r)
        elif t == "rook_endgame":
            pools["rook_endgame"].append(r)

    for cat, quota in nat_targets.items():
        pool = pools[cat]
        random.shuffle(pool)
        selected = pool[:quota]
        print(f"  Natural {cat:22s}: selected {len(selected)}/{quota} (pool: {len(pool)})")

        for r in selected:
            key = (r["fen"], r["move_uci"])
            if key in seen_fen_moves:
                continue
            seen_fen_moves.add(key)

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

    print(f"\nTotal assembled candidate positions: {len(analyses)}")
    records = [a.to_dict() for a in analyses]
    df = pd.DataFrame(records)

    # Strict deduplication assertion
    print("Asserting strict deduplication on (fen, move_uci)...")
    df_dedup = df.drop_duplicates(subset=["fen", "move_uci"]).reset_index(drop=True)
    print(f"Total rows: {len(df)} | Strictly unique (fen, move_uci): {len(df_dedup)}")
    assert len(df) == len(df_dedup), f"Duplicate (fen, move_uci) found: {len(df) - len(df_dedup)}"

    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Saved dataset to {OUTPUT_PARQUET}")

    print("\nTaxonomy Distribution:")
    print(df["tactic_type"].value_counts())
    return df

if __name__ == "__main__":
    build_hybrid_3000_dataset()
