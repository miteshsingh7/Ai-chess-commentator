"""Shared constants for the chess analysis pipeline.

Adapted from chessIQ constants without Streamlit/UI dependencies.
"""

from typing import Dict, Final
import chess

# Material values in standard centipawn equivalents / pawns
PIECE_VALUE: Final[Dict[chess.PieceType, int]] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

PIECE_NAMES: Final[Dict[chess.PieceType, str]] = {
    chess.QUEEN: "queen",
    chess.ROOK: "rook",
    chess.BISHOP: "bishop",
    chess.KNIGHT: "knight",
    chess.PAWN: "pawn",
    chess.KING: "king",
}

# Maximum centipawn cap: anything beyond +-900 is effectively decisive/mate
CP_CAP: Final[int] = 900

# Centipawn loss classification thresholds
GOOD_THRESHOLD: Final[int] = 50
INACCURACY_THRESHOLD: Final[int] = 100
MISTAKE_THRESHOLD: Final[int] = 300

# Canonical taxonomy labels recognized across the pipeline
VALID_TAXONOMY_TYPES: Final[tuple[str, ...]] = (
    "checkmate",
    "fork",
    "knight_fork",
    "pawn_fork",
    "pin",
    "skewer",
    "discovered_attack",
    "back_rank_mate",
    "weak_back_rank",
    "hanging_piece",
    "trapped_piece",
    "trapped_queen",
    "trapped_rook",
    "trapped_bishop",
    "trapped_knight",
    "missed_mate",
    "missed_pawn_fork",
    "overloaded_piece",
    "zwischenzug",
    "missed_sacrifice",
    "opening_calculation",
    "middlegame_calculation",
    "endgame_calculation",
    "calculation_error",
    "time_pressure",
    "king_exposed",
    "opening_principle",
    "rook_endgame",
    "pawn_endgame",
    "endgame_technique",
    "weak_pawns",
    "none",
)
