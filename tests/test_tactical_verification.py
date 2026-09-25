"""Unit tests for tactical verification elements and prompt injection (executed-tactic categories)."""

import pytest
import chess
from chess_commentator.analysis.taxonomy import (
    extract_fork_elements,
    extract_pin_elements,
    extract_skewer_elements,
    extract_discovered_elements,
    extract_trapped_elements,
    classify_position_taxonomy,
)
from chess_commentator.teacher.prompt_builder import (
    CONTINUATION_TACTIC_TYPES,
    VERIFICATION_TACTIC_TYPES,
    compute_tactical_verification,
    compute_tactical_continuation,
)
from chess_commentator.dataset.formatter import construct_user_prompt
from chess_commentator.serving.service import construct_serving_prompt
from chess_commentator.analysis.models import PositionAnalysis, TaxonomyResult, BoardFeatures


def test_extract_fork_elements(knight_fork_position):
    board, move = knight_fork_position  # d5c7 fork on King e8 and Rook a8
    elements = extract_fork_elements(board, move)
    assert elements is not None
    assert elements["forking_piece"] == "Knight on c7"
    assert "King on e8" in elements["target_pieces"]
    assert "Rook on a8" in elements["target_pieces"]


def test_extract_pin_elements():
    # White Bishop on a2, Black Queen on c4, Black King on g8 (diagonal a2-g8)
    fen = "6k1/8/8/8/2q5/8/B7/6K1 w - - 0 1"
    board = chess.Board(fen)
    # Move bishop from b1 to a2 (pinning queen on c4 to king on g8)
    fen_before = "6k1/8/8/8/2q5/8/8/B5K1 w - - 0 1"
    b_before = chess.Board(fen_before)
    move = chess.Move.from_uci("a1a2")
    elements = extract_pin_elements(b_before, move)
    assert elements is not None
    assert elements["pinning_piece"] == "Bishop on a2"
    assert elements["pinned_piece"] == "Queen on c4"
    assert elements["shielded_piece"] == "King on g8"


def test_extract_skewer_elements():
    # White Rook moves to e1 skewers King on e7 to Queen on e8
    fen = "4q3/4k3/8/8/8/8/8/R5K1 w - - 0 1"
    board = chess.Board(fen)
    move = chess.Move.from_uci("a1e1")
    elements = extract_skewer_elements(board, move)
    assert elements is not None
    assert elements["attacking_piece"] == "Rook on e1"
    assert elements["front_piece"] == "King on e7"
    assert elements["back_piece"] == "Queen on e8"


def test_extract_discovered_elements():
    # White Rook on d1, White Knight on d4, Black Queen on d8
    # Knight moves d4f5, unmasking Rook on d1 attacking Queen on d8
    fen = "3q4/8/8/8/3N4/8/8/3R2K1 w - - 0 1"
    board = chess.Board(fen)
    move = chess.Move.from_uci("d4f5")
    elements = extract_discovered_elements(board, move)
    assert elements is not None
    assert "Knight to f5" in elements["moved_piece"]
    assert elements["revealed_attacker"] == "Rook on d1"
    assert elements["target_piece"] == "Queen on d8"


def test_extract_trapped_elements():
    # Black Bishop on a7 with no safe moves, attacked/constrained by White pawn on b6 defended by a5, b8 covered by c7
    fen = "8/b1P5/1P6/P7/8/8/8/4K2k b - - 0 1"
    board = chess.Board(fen)
    # Black plays king move h1g1, bishop remains trapped on a7
    move = chess.Move.from_uci("h1g1")
    elements = extract_trapped_elements(board, move, chess.BLACK)
    assert elements is not None
    assert elements["trapped_piece"] == "Bishop on a7"
    assert any("b6" in p for p in elements["constraining_pieces"])


def test_compute_tactical_verification_fork(knight_fork_position):
    board, move = knight_fork_position
    fen = board.fen()
    uci = move.uci()
    block = compute_tactical_verification(
        fen=fen,
        move_uci=uci,
        tactic="knight_fork",
        player_color="white",
    )
    assert block is not None
    assert "Tactical Elements:" in block
    assert "- Forking Piece: Knight on c7" in block
    assert "King on e8" in block
    assert "Rook on a8" in block
    assert "Base your explanation only on the position, the move played, and the tactical elements shown above" in block


def test_construct_user_prompt_dispatches_correctly(knight_fork_position):
    board, move = knight_fork_position
    fen = board.fen()
    uci = move.uci()

    # 1. Executed tactic (knight_fork) gets Tactical Elements
    fork_prompt = construct_user_prompt(
        fen=fen,
        move_san="Nc7+",
        move_uci=uci,
        player_color="white",
        mistake_type="good",
        cp_loss=0,
        tactic_type="knight_fork",
    )
    assert "Tactical Elements:" in fork_prompt
    assert "Forking Piece: Knight on c7" in fork_prompt
    assert "Tactical Continuation:" not in fork_prompt

    # 2. Continuation tactic (hanging_piece) gets Tactical Continuation
    hanging_prompt = construct_user_prompt(
        fen=fen,
        move_san="Nc7+",
        move_uci=uci,
        player_color="white",
        mistake_type="mistake",
        cp_loss=280,
        best_move="g1f2",
        tactic_type="hanging_piece",
    )
    assert "Tactical Continuation:" in hanging_prompt
    assert "Tactical Elements:" not in hanging_prompt

    # 3. None tactic gets standard prompt without extra block
    none_prompt = construct_user_prompt(
        fen=fen,
        move_san="Nc7+",
        move_uci=uci,
        player_color="white",
        mistake_type="good",
        cp_loss=0,
        tactic_type="none",
    )
    assert "Tactical Continuation:" not in none_prompt
    assert "Tactical Elements:" not in none_prompt
    assert "Base your explanation" not in none_prompt


def test_serving_prompt_parity_for_verification_tactic(knight_fork_position):
    board, move = knight_fork_position
    analysis = PositionAnalysis(
        fen=board.fen(),
        move_san="Nc7+",
        move_uci=move.uci(),
        player_color="white",
        eval_before=100,
        eval_after=400,
        cp_loss=0,
        mistake_type="good",
        best_move=move.uci(),
        played_best=True,
        taxonomy=TaxonomyResult(
            mistake_category="none",
            tactic_type="knight_fork",
        ),
        features=BoardFeatures(
            phase="endgame",
            castled=False,
            open_files_near_king=0,
            doubled_pawns=0,
            isolated_pawns=0,
            passed_pawns=0,
            mobility=10,
            time_pressure=False,
            pawns=0,
            knights=1,
            bishops=0,
            rooks=0,
            queens=0,
            opp_pawns=0,
            opp_knights=0,
            opp_bishops=0,
            opp_rooks=1,
            opp_queens=0,
            material_balance=3,
        ),
        move_number=30,
    )

    serving_prompt = construct_serving_prompt(analysis)
    assert "Tactical Elements:" in serving_prompt
    assert "Forking Piece: Knight on c7" in serving_prompt
    assert "Base your explanation only on the position, the move played, and the tactical elements shown above" in serving_prompt
