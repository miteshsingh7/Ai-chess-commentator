"""Unit tests for feature extraction."""

import chess
from chess_commentator.analysis.features import (
    get_total_material,
    get_game_phase,
    is_castled,
    count_open_files_near_king,
    count_doubled_pawns,
    count_isolated_pawns,
    count_passed_pawns,
    get_mobility,
    extract_board_features,
)


def test_starting_board_material(starting_board):
    # Total piece value in standard setup: 2*(8*1 + 2*3 + 2*3 + 2*5 + 9) = 2*(8+6+6+10+9) = 2*39 = 78
    assert get_total_material(starting_board) == 78


def test_phase_detection(starting_board):
    # Move number <= 12 should be opening
    assert get_game_phase(starting_board, move_number=5) == "opening"
    # Move 15 with full material is middlegame
    assert get_game_phase(starting_board, move_number=15) == "middlegame"

    # Endgame with low material
    endgame_board = chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1")
    assert get_game_phase(endgame_board, move_number=40) == "endgame"


def test_is_castled():
    # White King on g1 is castled kingside
    board_castled = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4")
    assert is_castled(board_castled, chess.WHITE) is True

    # Starting position: king on e1 is not castled
    board_start = chess.Board()
    assert is_castled(board_start, chess.WHITE) is False


def test_pawn_structure_metrics():
    # Board with doubled white pawns on c-file (c2, c3) and isolated pawn on a3 (no b-pawn)
    fen = "8/8/8/8/8/P1P5/2P5/4K2k w - - 0 1"
    board = chess.Board(fen)
    assert count_doubled_pawns(board, chess.WHITE) == 1
    assert count_isolated_pawns(board, chess.WHITE) >= 1


def test_passed_pawn():
    # White passed pawn on d6 with no black pawns blocking
    fen = "4k3/8/3P4/8/8/8/8/4K3 w - - 0 1"
    board = chess.Board(fen)
    assert count_passed_pawns(board, chess.WHITE) == 1


def test_mobility_and_features_model(starting_board):
    features = extract_board_features(starting_board, chess.WHITE, move_number=1)
    assert features.phase == "opening"
    assert features.mobility == 20  # 16 pawn moves + 4 knight moves
    assert features.material_balance == 78
    assert features.castled is False
