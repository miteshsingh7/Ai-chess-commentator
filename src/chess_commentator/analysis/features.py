"""Pure functions for extracting positional, king safety, and structural chess features.

Extracted and refined from chessIQ phase3_feature_engineering.py.
"""

from typing import Dict, Optional, Set
import chess
from chess_commentator.analysis.constants import PIECE_VALUE
from chess_commentator.analysis.models import BoardFeatures


def get_total_material(board: chess.Board) -> int:
    """Calculate total non-king piece material remaining on the board."""
    total: int = 0
    for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        val = PIECE_VALUE.get(piece_type, 0)
        total += (white_count + black_count) * val
    return total


def get_game_phase(board: chess.Board, move_number: int) -> str:
    """Determine phase: opening, middlegame, or endgame."""
    has_white_queen: bool = bool(board.pieces(chess.QUEEN, chess.WHITE))
    has_black_queen: bool = bool(board.pieces(chess.QUEEN, chess.BLACK))
    total_material: int = get_total_material(board)

    if move_number <= 12:
        return "opening"
    if total_material <= 20 or (not has_white_queen and not has_black_queen):
        return "endgame"
    return "middlegame"


def is_castled(board: chess.Board, color: chess.Color) -> bool:
    """Heuristic check: king has castled away from central files."""
    king_sq: Optional[int] = board.king(color)
    if king_sq is None:
        return False
    king_file: int = chess.square_file(king_sq)
    # Kingside castle: file g (6), Queenside castle: file c (2) or b (1)
    return king_file in (1, 2, 6, 7)


def count_open_files_near_king(board: chess.Board, color: chess.Color) -> int:
    """Count adjacent files to the king that have no pawns (higher = more danger)."""
    king_sq: Optional[int] = board.king(color)
    if king_sq is None:
        return 0
    king_file: int = chess.square_file(king_sq)
    open_files: int = 0
    for f in range(max(0, king_file - 1), min(8, king_file + 2)):
        white_pawns = any(
            chess.square_file(sq) == f for sq in board.pieces(chess.PAWN, chess.WHITE)
        )
        black_pawns = any(
            chess.square_file(sq) == f for sq in board.pieces(chess.PAWN, chess.BLACK)
        )
        if not white_pawns and not black_pawns:
            open_files += 1
    return open_files


def count_doubled_pawns(board: chess.Board, color: chess.Color) -> int:
    """Count pawns sharing the same file for the given color."""
    doubled: int = 0
    for file_idx in range(8):
        pawns_on_file = sum(
            1 for sq in board.pieces(chess.PAWN, color)
            if chess.square_file(sq) == file_idx
        )
        if pawns_on_file > 1:
            doubled += (pawns_on_file - 1)
    return doubled


def count_isolated_pawns(board: chess.Board, color: chess.Color) -> int:
    """Count pawns that have no friendly pawns on adjacent files."""
    isolated: int = 0
    pawn_files: Set[int] = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)}
    for f in pawn_files:
        neighbors = {f - 1, f + 1} & pawn_files
        if not neighbors:
            isolated += 1
    return isolated


def count_passed_pawns(board: chess.Board, color: chess.Color) -> int:
    """Count pawns with no opposing pawns in front on the same or adjacent files."""
    passed: int = 0
    opponent: chess.Color = not color
    opp_pawns = list(board.pieces(chess.PAWN, opponent))

    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        blocking_files = {f - 1, f, f + 1} & set(range(8))
        is_passed = True
        for opp_sq in opp_pawns:
            opp_f = chess.square_file(opp_sq)
            opp_r = chess.square_rank(opp_sq)
            if opp_f in blocking_files:
                if color == chess.WHITE and opp_r > r:
                    is_passed = False
                    break
                if color == chess.BLACK and opp_r < r:
                    is_passed = False
                    break
        if is_passed:
            passed += 1
    return passed


def get_mobility(board: chess.Board, color: chess.Color) -> int:
    """Count number of legal moves available for the given color."""
    board_copy = board.copy()
    board_copy.turn = color
    return board_copy.legal_moves.count()


def get_material_features(board: chess.Board, color: chess.Color) -> Dict[str, int]:
    """Return piece counts for player and opponent, plus total material balance."""
    opp: chess.Color = not color
    return {
        "pawns": len(board.pieces(chess.PAWN, color)),
        "knights": len(board.pieces(chess.KNIGHT, color)),
        "bishops": len(board.pieces(chess.BISHOP, color)),
        "rooks": len(board.pieces(chess.ROOK, color)),
        "queens": len(board.pieces(chess.QUEEN, color)),
        "opp_pawns": len(board.pieces(chess.PAWN, opp)),
        "opp_knights": len(board.pieces(chess.KNIGHT, opp)),
        "opp_bishops": len(board.pieces(chess.BISHOP, opp)),
        "opp_rooks": len(board.pieces(chess.ROOK, opp)),
        "opp_queens": len(board.pieces(chess.QUEEN, opp)),
        "material_balance": get_total_material(board),
    }


def extract_board_features(
    board: chess.Board,
    color: chess.Color,
    move_number: int = 20,
    time_left: Optional[int] = None,
) -> BoardFeatures:
    """Extract complete positional features from a chess.Board state."""
    material = get_material_features(board, color)
    phase = get_game_phase(board, move_number)
    castled = is_castled(board, color)
    open_files = count_open_files_near_king(board, color)
    doubled = count_doubled_pawns(board, color)
    isolated = count_isolated_pawns(board, color)
    passed = count_passed_pawns(board, color)
    mobility = get_mobility(board, color)
    time_pressure = (time_left is not None and time_left < 60)

    return BoardFeatures(
        phase=phase,
        castled=castled,
        open_files_near_king=open_files,
        doubled_pawns=doubled,
        isolated_pawns=isolated,
        passed_pawns=passed,
        mobility=mobility,
        time_pressure=time_pressure,
        pawns=material["pawns"],
        knights=material["knights"],
        bishops=material["bishops"],
        rooks=material["rooks"],
        queens=material["queens"],
        opp_pawns=material["opp_pawns"],
        opp_knights=material["opp_knights"],
        opp_bishops=material["opp_bishops"],
        opp_rooks=material["opp_rooks"],
        opp_queens=material["opp_queens"],
        material_balance=material["material_balance"],
    )
