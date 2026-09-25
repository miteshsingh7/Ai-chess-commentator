"""Test fixtures and sample tactical positions."""

import pytest
import chess


@pytest.fixture
def starting_board():
    return chess.Board()


@pytest.fixture
def knight_fork_position():
    """Position where White knight can fork Black king on e8 and rook on a8 via Nc7+."""
    # White King on g1, Knight on d5. Black King on e8, Rook on a8.
    fen = "r3k3/8/8/3N4/8/8/8/6K1 w - - 0 1"
    return chess.Board(fen), chess.Move.from_uci("d5c7")


@pytest.fixture
def pin_position():
    """Position where White Bishop pins Black Queen to King."""
    # White Bishop on c4, Black Queen on e6, Black King on g8
    # Diagonal c4-d5-e6-f7-g8
    fen = "6k1/5p2/4q3/8/2B5/8/8/6K1 w - - 0 1"
    return chess.Board(fen), chess.Move.from_uci("c4e6")


@pytest.fixture
def hanging_piece_position():
    """Position where White plays move leaving Knight on e5 hanging."""
    # White Queen on d1, Knight on f3 moves to e5 where Black d6 pawn can take it for free
    fen = "r1bqkbnr/ppp1pppp/3p4/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 3"
    return chess.Board(fen), chess.Move.from_uci("f3e5")


@pytest.fixture
def back_rank_mate_position():
    """Position where White plays Ra8# executing a back rank checkmate."""
    fen = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
    return chess.Board(fen), chess.Move.from_uci("a1a8")


@pytest.fixture
def sample_pgn_text():
    return """[Event "Casual Game"]
[Site "Chess.com"]
[Date "2024.01.01"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1480"]
[ECO "C20"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
"""
