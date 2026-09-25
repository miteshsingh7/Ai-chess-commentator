"""Sample and stratify exactly 50 positions from chessIQ data for the pilot generation."""

import os
import sys
import chess
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from chess_commentator.analysis.taxonomy import classify_position_taxonomy
from chess_commentator.analysis.models import BoardFeatures


def main() -> None:
    source_path = "/Users/miteshsingh/Documents/projects/chess_analyzer/data/processed/moves_categorized.parquet"
    if not os.path.exists(source_path):
        print(f"Error: {source_path} not found.")
        sys.exit(1)

    df = pd.read_parquet(source_path)
    print(f"Loaded {len(df)} total moves from chessIQ dataset.")

    # Re-classify with updated taxonomy.py so checkmate and positive tactics are identified
    records = df.to_dict("records")
    updated_records = []

    for r in records:
        board = chess.Board(r["fen"])
        move = chess.Move.from_uci(r["move_uci"])
        color = chess.WHITE if r["player_color"] == "white" else chess.BLACK
        best_uci = r.get("best_move")
        best_move = chess.Move.from_uci(best_uci) if best_uci else None
        loss = r.get("cp_loss")
        eval_before = r.get("eval_before")
        eval_after = r.get("eval_after")
        played_best = bool(r.get("played_best", False))

        tax = classify_position_taxonomy(
            board=board,
            move=move,
            player_color=color,
            eval_before=eval_before,
            eval_after=eval_after,
            cp_loss=loss,
            best_move=best_move,
            played_best=played_best,
        )

        r["mistake_category"] = tax.mistake_category
        r["tactic_type"] = tax.tactic_type
        r["piece_lost"] = tax.piece_lost
        r["mate_missed"] = tax.mate_missed
        updated_records.append(r)

    df_updated = pd.DataFrame(updated_records)
    print("\nUpdated Tactic Type Distribution:")
    print(df_updated["tactic_type"].value_counts().head(15))

    # Stratified sampling of exactly 50 positions:
    # 1. Checkmate (5)
    # If not enough in df, we will scan board_after.is_checkmate()
    checkmates = df_updated[df_updated["tactic_type"] == "checkmate"]
    if len(checkmates) < 5:
        # Also check PGN or known scholar's / back-rank checkmates if needed
        pass

    hanging = df_updated[df_updated["tactic_type"] == "hanging_piece"]
    forks = df_updated[df_updated["tactic_type"].isin(["knight_fork", "fork", "pawn_fork"])]
    missed_mates_sac = df_updated[df_updated["tactic_type"].isin(["missed_mate", "missed_sacrifice"])]
    pins_skewers = df_updated[df_updated["tactic_type"].isin(["pin", "skewer", "discovered_attack", "trapped_knight", "trapped_bishop", "trapped_rook"])]
    overloaded_zw = df_updated[df_updated["tactic_type"].isin(["overloaded_piece", "zwischenzug", "back_rank_mate"])]
    quiet_good = df_updated[(df_updated["mistake_type"] == "good") & (df_updated["tactic_type"] == "none")]

    selected_dfs = []
    
    # Checkmates: take up to 5
    selected_dfs.append(checkmates.head(5))
    # Hanging pieces: 8
    selected_dfs.append(hanging.head(8))
    # Forks: 8
    selected_dfs.append(forks.head(8))
    # Missed mates & sacrifices: 6
    selected_dfs.append(missed_mates_sac.head(6))
    # Pins & skewers / trapped: 6
    selected_dfs.append(pins_skewers.head(6))
    # Overloaded / zwischenzug: 5
    selected_dfs.append(overloaded_zw.head(5))
    
    current_count = sum(len(d) for d in selected_dfs)
    remaining_needed = 50 - current_count
    # Quiet good moves: remaining
    selected_dfs.append(quiet_good.head(remaining_needed))

    pilot_df = pd.concat(selected_dfs, ignore_index=True)
    
    # If checkmates were fewer than 4 in the existing 1823 rapid moves, add a couple canonical checkmates
    if len(pilot_df[pilot_df["tactic_type"] == "checkmate"]) < 3:
        # Add scholar's mate and back-rank mate positions
        canonical_mates = [
            {
                "game_id": "canonical_scholars_mate",
                "move_number": 4,
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
                "move_san": "Qxf7#",
                "move_uci": "h5f7",
                "player_color": "white",
                "result": "win",
                "time_control": "600",
                "eco": "C20",
                "opening": "Scholar's Mate",
                "white_player": "White",
                "black_player": "Black",
                "eval_before": 900,
                "eval_after": 900,
                "cp_loss": 0,
                "best_move": "h5f7",
                "played_best": True,
                "mistake_type": "good",
                "phase": "opening",
                "castled": False,
                "open_files_near_king": 0,
                "doubled_pawns": 0,
                "isolated_pawns": 0,
                "passed_pawns": 0,
                "mobility": 20,
                "time_pressure": False,
                "pawns": 8, "knights": 2, "bishops": 2, "rooks": 2, "queens": 1,
                "opp_pawns": 7, "opp_knights": 2, "opp_bishops": 1, "opp_rooks": 2, "opp_queens": 1,
                "material_balance": 77,
                "mistake_category": "none",
                "tactic_type": "checkmate",
                "piece_lost": "none",
                "mate_missed": 0,
            },
            {
                "game_id": "canonical_back_rank_mate",
                "move_number": 28,
                "fen": "4r1k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
                "move_san": "Ra8",
                "move_uci": "a1a8",
                "player_color": "white",
                "result": "win",
                "time_control": "600",
                "eco": "E00",
                "opening": "Endgame",
                "white_player": "White",
                "black_player": "Black",
                "eval_before": 900,
                "eval_after": 900,
                "cp_loss": 0,
                "best_move": "a1a8",
                "played_best": True,
                "mistake_type": "good",
                "phase": "endgame",
                "castled": True,
                "open_files_near_king": 0,
                "doubled_pawns": 0,
                "isolated_pawns": 0,
                "passed_pawns": 0,
                "mobility": 14,
                "time_pressure": False,
                "pawns": 3, "knights": 0, "bishops": 0, "rooks": 1, "queens": 0,
                "opp_pawns": 3, "opp_knights": 0, "opp_bishops": 0, "opp_rooks": 1, "opp_queens": 0,
                "material_balance": 16,
                "mistake_category": "none",
                "tactic_type": "back_rank_mate",
                "piece_lost": "none",
                "mate_missed": 0,
            },
            {
                "game_id": "canonical_fools_mate",
                "move_number": 2,
                "fen": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR b KQkq - 1 2",
                "move_san": "Qh4#",
                "move_uci": "d8h4",
                "player_color": "black",
                "result": "win",
                "time_control": "600",
                "eco": "B00",
                "opening": "Fool's Mate",
                "white_player": "White",
                "black_player": "Black",
                "eval_before": -900,
                "eval_after": -900,
                "cp_loss": 0,
                "best_move": "d8h4",
                "played_best": True,
                "mistake_type": "good",
                "phase": "opening",
                "castled": False,
                "open_files_near_king": 1,
                "doubled_pawns": 0,
                "isolated_pawns": 0,
                "passed_pawns": 0,
                "mobility": 20,
                "time_pressure": False,
                "pawns": 7, "knights": 2, "bishops": 2, "rooks": 2, "queens": 1,
                "opp_pawns": 7, "opp_knights": 2, "opp_bishops": 2, "opp_rooks": 2, "opp_queens": 1,
                "material_balance": 76,
                "mistake_category": "none",
                "tactic_type": "checkmate",
                "piece_lost": "none",
                "mate_missed": 0,
            }
        ]
        pilot_df = pd.concat([pd.DataFrame(canonical_mates), pilot_df], ignore_index=True)

    pilot_df = pilot_df.drop_duplicates(subset=["fen", "move_uci"]).head(50).reset_index(drop=True)
    os.makedirs("data", exist_ok=True)
    output_path = "data/pilot_50_positions.parquet"
    pilot_df.to_parquet(output_path, index=False)

    print(f"\nFinal Pilot Set (exactly {len(pilot_df)} positions) saved to {output_path}")
    print("\nTactic distribution in Pilot 50:")
    print(pilot_df["tactic_type"].value_counts())
    print("\nMistake distribution in Pilot 50:")
    print(pilot_df["mistake_type"].value_counts())


if __name__ == "__main__":
    main()
