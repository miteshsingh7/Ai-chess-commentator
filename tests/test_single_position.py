"""Integration and unit tests for single position analyzer and PGN parser."""

import chess
import pytest
from chess_commentator.analysis.pipeline import analyze_position, parse_move_on_board
from chess_commentator.analysis.parser import parse_pgn_string
from chess_commentator.analysis.models import PositionAnalysis


def test_parse_move_on_board(starting_board):
    # UCI
    move_uci = parse_move_on_board(starting_board, "e2e4")
    assert move_uci == chess.Move.from_uci("e2e4")

    # SAN
    move_san = parse_move_on_board(starting_board, "Nf3")
    assert move_san == chess.Move.from_uci("g1f3")

    # Illegal move raises ValueError
    with pytest.raises(ValueError):
        parse_move_on_board(starting_board, "e5")


def test_parse_pgn_string(sample_pgn_text):
    rows = parse_pgn_string(sample_pgn_text)
    assert len(rows) == 7  # Scholar's mate has 7 half-moves
    assert rows[0]["move_san"] == "e4"
    assert rows[-1]["move_san"] == "Qxf7#"


def test_analyze_position_end_to_end():
    # Scholar's mate final position before mate
    # Black just played Nf6?? allowing Qxf7#
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    analysis: PositionAnalysis = analyze_position(
        fen=fen,
        move="Qxf7#",
        depth=10,  # fast depth for test speed
    )

    assert analysis.move_san == "Qxf7#"
    assert analysis.move_uci == "h5f7"
    assert analysis.player_color == "white"
    assert analysis.played_best is True
    assert analysis.mistake_type == "good"
    assert analysis.taxonomy.mistake_category == "none"
    assert analysis.taxonomy.tactic_type == "checkmate"
    assert analysis.features.material_balance > 0
