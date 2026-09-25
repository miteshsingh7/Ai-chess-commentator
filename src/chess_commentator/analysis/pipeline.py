"""Unified analysis pipeline for single positions and full PGN files."""

import os
from typing import Optional, List, Union
import chess
import pandas as pd
from tqdm import tqdm

from chess_commentator.analysis.engine import StockfishEngine
from chess_commentator.analysis.features import extract_board_features
from chess_commentator.analysis.models import (
    PositionAnalysis,
    MoveMetadata,
    BoardFeatures,
    TaxonomyResult,
)
from chess_commentator.analysis.parser import parse_pgn_file, parse_pgn_string
from chess_commentator.analysis.taxonomy import classify_position_taxonomy


def parse_move_on_board(board: chess.Board, move_str: str) -> chess.Move:
    """Parse a move string given as either UCI (e.g. 'e2e4') or SAN (e.g. 'e4')."""
    clean_move = move_str.strip()
    try:
        move = chess.Move.from_uci(clean_move)
        if move in board.legal_moves:
            return move
    except ValueError:
        pass

    try:
        return board.parse_san(clean_move)
    except ValueError as e:
        raise ValueError(
            f"Move '{move_str}' is neither a valid UCI nor legal SAN in position: {board.fen()}"
        ) from e


def analyze_position(
    fen: str,
    move: Union[str, chess.Move],
    engine: Optional[StockfishEngine] = None,
    depth: int = 18,
    time_limit: Optional[float] = None,
    mode: str = "deep",
    move_number: int = 20,
    time_left: Optional[int] = None,
    metadata: Optional[MoveMetadata] = None,
) -> PositionAnalysis:
    """Analyze a single chess position and move, returning grounded evaluation and taxonomy."""
    board = chess.Board(fen)
    chess_move = move if isinstance(move, chess.Move) else parse_move_on_board(board, move)
    san = board.san(chess_move)
    uci = chess_move.uci()
    player_color = board.turn
    color_str = "white" if player_color == chess.WHITE else "black"

    # Extract board features
    features: BoardFeatures = extract_board_features(
        board=board,
        color=player_color,
        move_number=move_number,
        time_left=time_left,
    )

    # Engine evaluation
    local_engine = engine or StockfishEngine(default_depth=depth)
    should_close = engine is None

    try:
        eval_data = local_engine.evaluate_move(
            board=board,
            move=chess_move,
            depth=depth,
            time_limit=time_limit,
        )
    finally:
        if should_close:
            local_engine.close()

    eval_before = eval_data["eval_before"]
    eval_after = eval_data["eval_after"]
    cp_loss = eval_data["cp_loss"]
    best_uci = eval_data["best_move"]
    played_best = eval_data["played_best"]
    mistake_type = eval_data["mistake_type"]
    best_move_obj = chess.Move.from_uci(best_uci) if best_uci else None

    # Tactical taxonomy classification
    taxonomy: TaxonomyResult = classify_position_taxonomy(
        board=board,
        move=chess_move,
        player_color=player_color,
        eval_before=eval_before,
        eval_after=eval_after,
        cp_loss=cp_loss,
        best_move=best_move_obj,
        played_best=played_best,
        mode=mode,
        features=features,
        move_number=move_number,
    )

    return PositionAnalysis(
        fen=fen,
        move_uci=uci,
        move_san=san,
        player_color=color_str,
        move_number=move_number,
        eval_before=eval_before,
        eval_after=eval_after,
        cp_loss=cp_loss,
        best_move=best_uci,
        played_best=played_best,
        mistake_type=mistake_type,
        taxonomy=taxonomy,
        features=features,
        metadata=metadata,
    )


class AnalysisPipeline:
    """Orchestrator for analyzing batches of moves and games."""

    def __init__(
        self,
        engine: Optional[StockfishEngine] = None,
        depth: int = 18,
        time_limit: Optional[float] = None,
        mode: str = "deep",
    ) -> None:
        self.engine = engine or StockfishEngine(default_depth=depth)
        self.depth = depth
        self.time_limit = time_limit
        self.mode = mode

    def analyze_pgn(
        self,
        pgn_path_or_text: str,
        player_username: Optional[str] = None,
        max_moves: Optional[int] = None,
    ) -> List[PositionAnalysis]:
        """Analyze moves from a PGN file path or raw PGN string."""
        if os.path.exists(pgn_path_or_text):
            move_records = parse_pgn_file(pgn_path_or_text, player_username)
        else:
            move_records = parse_pgn_string(pgn_path_or_text, player_username)

        if max_moves is not None:
            move_records = move_records[:max_moves]

        results: List[PositionAnalysis] = []
        for rec in tqdm(move_records, desc="Analyzing moves"):
            meta = MoveMetadata(
                game_id=rec["game_id"],
                move_number=rec["move_number"],
                player_color=rec["player_color"],
                move_san=rec["move_san"],
                move_uci=rec["move_uci"],
                time_left=rec.get("time_left"),
                result=rec.get("result", "unknown"),
                eco=rec.get("eco", "?"),
                opening=rec.get("opening", "?"),
                white_player=rec.get("white_player", ""),
                black_player=rec.get("black_player", ""),
                white_elo=rec.get("white_elo"),
                black_elo=rec.get("black_elo"),
            )

            analysis = analyze_position(
                fen=rec["fen"],
                move=rec["move_uci"],
                engine=self.engine,
                depth=self.depth,
                time_limit=self.time_limit,
                mode=self.mode,
                move_number=rec["move_number"],
                time_left=rec.get("time_left"),
                metadata=meta,
            )
            results.append(analysis)

        return results

    def analyze_to_dataframe(
        self,
        pgn_path_or_text: str,
        player_username: Optional[str] = None,
        output_parquet: Optional[str] = None,
        max_moves: Optional[int] = None,
    ) -> pd.DataFrame:
        """Analyze moves and return as a Pandas DataFrame, optionally saving to Parquet."""
        analyses = self.analyze_pgn(pgn_path_or_text, player_username, max_moves)
        data = [a.to_dict() for a in analyses]
        df = pd.DataFrame(data)

        if output_parquet and not df.empty:
            dir_name = os.path.dirname(output_parquet)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            df.to_parquet(output_parquet, index=False)

        return df


def analyze_pgn_file(
    pgn_path: str,
    output_parquet: Optional[str] = None,
    player_username: Optional[str] = None,
    depth: int = 18,
    mode: str = "deep",
) -> pd.DataFrame:
    """Convenience helper to analyze a PGN file directly to Parquet."""
    pipeline = AnalysisPipeline(depth=depth, mode=mode)
    try:
        return pipeline.analyze_to_dataframe(
            pgn_path_or_text=pgn_path,
            player_username=player_username,
            output_parquet=output_parquet,
        )
    finally:
        pipeline.engine.close()
