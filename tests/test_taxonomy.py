"""Unit tests for tactical taxonomy detectors and classifier."""

import pytest
import chess
from chess_commentator.analysis.taxonomy import (
    detect_fork,
    detect_pin,
    detect_skewer,
    detect_back_rank,
    detect_hanging,
    detect_discovered,
    detect_trapped_piece,
    detect_overloaded_piece,
    detect_sacrifice_missed,
    detect_zwischenzug,
    classify_position_taxonomy,
)


def test_detect_knight_fork(knight_fork_position):
    board, move = knight_fork_position
    assert detect_fork(board, move) is True


def test_detect_back_rank(back_rank_mate_position):
    board, move = back_rank_mate_position
    assert detect_back_rank(board, move) is True


def test_detect_skewer():
    # White Rook on e1, Black King on e7, Black Queen on e8
    # When Rook gives check from e1, king must move, exposing queen
    fen = "4q3/4k3/8/8/8/8/8/4R1K1 w - - 0 1"
    board = chess.Board(fen)
    # The rook is on e1 already; let's test a move from a1 to e1
    fen2 = "4q3/4k3/8/8/8/8/8/R5K1 w - - 0 1"
    b2 = chess.Board(fen2)
    skewer_move = chess.Move.from_uci("a1e1")
    assert detect_skewer(b2, skewer_move) is True


def test_detect_hanging(hanging_piece_position):
    board, move = hanging_piece_position
    is_hang, hung_piece = detect_hanging(board, move, chess.WHITE)
    assert is_hang is True
    assert hung_piece == chess.KNIGHT


def test_classify_hanging_piece_taxonomy(hanging_piece_position):
    board, move = hanging_piece_position
    result = classify_position_taxonomy(
        board=board,
        move=move,
        player_color=chess.WHITE,
        eval_before=50,
        eval_after=-250,
        cp_loss=300,
        best_move=chess.Move.from_uci("d2d4"),
        played_best=False,
    )
    assert result.mistake_category == "hanging_piece"
    assert result.tactic_type == "hanging_piece"
    assert result.piece_lost == "knight"


def test_classify_missed_fork_taxonomy(knight_fork_position):
    board, fork_move = knight_fork_position
    # Player played a quiet move like Kh1 instead of the fork Nc7+
    quiet_move = chess.Move.from_uci("g1h1")
    result = classify_position_taxonomy(
        board=board,
        move=quiet_move,
        player_color=chess.WHITE,
        eval_before=500,
        eval_after=100,
        cp_loss=400,
        best_move=fork_move,
        played_best=False,
    )
    assert result.mistake_category == "missed_tactic"
    assert result.tactic_type == "knight_fork"


def test_detect_missed_mate_eval_threshold():
    """Assert missed mate operates strictly on centipawn thresholds without board-wide attack scans."""
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")

    # 1. Missed mate: eval_before >= 900 (CP_CAP) and cp_loss >= 200
    res_missed = classify_position_taxonomy(
        board=board,
        move=move,
        player_color=chess.WHITE,
        eval_before=900,
        eval_after=500,
        cp_loss=400,
        best_move=chess.Move.from_uci("d2d4"),
        played_best=False,
    )
    assert res_missed.mistake_category == "missed_mate"
    assert res_missed.tactic_type == "missed_mate"
    assert res_missed.mate_missed == 1

    # 2. Not missed mate: high cp_loss but eval_before < CP_CAP (regular blunder, not missed forced mate)
    res_not_mate = classify_position_taxonomy(
        board=board,
        move=move,
        player_color=chess.WHITE,
        eval_before=400,
        eval_after=-100,
        cp_loss=500,
        best_move=chess.Move.from_uci("d2d4"),
        played_best=False,
    )
    assert res_not_mate.tactic_type != "missed_mate"


def test_classify_good_move(starting_board):
    move = chess.Move.from_uci("e2e4")
    result = classify_position_taxonomy(
        board=starting_board,
        move=move,
        player_color=chess.WHITE,
        eval_before=20,
        eval_after=25,
        cp_loss=0,
        best_move=move,
        played_best=True,
    )
    assert result.mistake_category == "none"
    assert result.tactic_type == "none"


def test_classify_checkmate_regardless_of_cp_loss():
    """Assert a checkmating move is always classified as tactic_type='checkmate' regardless of cp_loss."""
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    board = chess.Board(fen)
    mate_move = chess.Move.from_uci("h5f7")

    # Test with cp_loss=0 (optimal play)
    res_zero = classify_position_taxonomy(
        board=board,
        move=mate_move,
        player_color=chess.WHITE,
        eval_before=900,
        eval_after=900,
        cp_loss=0,
        played_best=True,
    )
    assert res_zero.tactic_type == "checkmate"

    # Test with cp_loss=500 (anomaly / high cp_loss input)
    res_high_loss = classify_position_taxonomy(
        board=board,
        move=mate_move,
        player_color=chess.WHITE,
        eval_before=900,
        eval_after=400,
        cp_loss=500,
        played_best=False,
    )
    assert res_high_loss.tactic_type == "checkmate"

    # Test with cp_loss=None
    res_none_loss = classify_position_taxonomy(
        board=board,
        move=mate_move,
        player_color=chess.WHITE,
        eval_before=None,
        eval_after=None,
        cp_loss=None,
        played_best=False,
    )
    assert res_none_loss.tactic_type == "checkmate"


def test_classify_executed_tactics_on_good_move(knight_fork_position):
    """Assert a good move (cp_loss < 50) executing a tactic is classified with that motif instead of 'none'."""
    board, fork_move = knight_fork_position
    result = classify_position_taxonomy(
        board=board,
        move=fork_move,
        player_color=chess.WHITE,
        eval_before=500,
        eval_after=500,
        cp_loss=0,
        best_move=fork_move,
        played_best=True,
    )
    assert result.mistake_category == "none"
    assert result.tactic_type == "knight_fork"
    assert result.tactic_type != "none"


@pytest.fixture
def consecutive_good_moves_fixture():
    """Fixture with the exact 8-position sequence (positions 14-21) from the pilot run.
    
    Contains consecutive GOOD moves from games where previously a team-wide attack
    bug falsely classified every move as 'fork'.
    """
    return [
        {
            "game_id": "game_1",
            "move_number": 14,
            "fen": "rnbqkb1r/pp2ppp1/5n2/2P4p/7P/6N1/PPP2PP1/R1BQKBNR b KQkq - 0 7",
            "move_uci": "d8d1",
            "move_san": "Qxd1+",
            "player_color": chess.BLACK,
            "cp_loss": 2,
            "best_move": "d8d1",
            "played_best": True,
            "eval_before": 26,
            "eval_after": 24,
            "expected_tactic": "fork",
        },
        {
            "game_id": "game_1",
            "move_number": 34,
            "fen": "2k2b1r/pp1r2p1/4p3/2n3Np/7P/4N3/PPP1KPP1/R2R4 b - - 3 17",
            "move_uci": "f8e7",
            "move_san": "Be7",
            "player_color": chess.BLACK,
            "cp_loss": 28,
            "best_move": "d7c7",
            "played_best": False,
            "eval_before": -176,
            "eval_after": -204,
            "expected_tactic": "none",
        },
        {
            "game_id": "game_1",
            "move_number": 46,
            "fen": "8/ppk3p1/4p3/2n1N2p/1P5b/4N3/P1P2rP1/3RK3 b - - 1 23",
            "move_uci": "f2f5",
            "move_san": "Rf5+",
            "player_color": chess.BLACK,
            "cp_loss": 8,
            "best_move": "f2f5",
            "played_best": True,
            "eval_before": 417,
            "eval_after": 409,
            "expected_tactic": "discovered_attack",
        },
        {
            "game_id": "game_2",
            "move_number": 51,
            "fen": "3r1r1k/pp4pp/1q4b1/8/3bPp2/1B3P2/P5PP/1R1QRN1K w - - 2 26",
            "move_uci": "b3d5",
            "move_san": "Bd5",
            "player_color": chess.WHITE,
            "cp_loss": 0,
            "best_move": "b3d5",
            "played_best": True,
            "eval_before": -2,
            "eval_after": -2,
            "expected_tactic": "discovered_attack",
        },
        {
            "game_id": "game_3",
            "move_number": 50,
            "fen": "r2qrnk1/5p2/2p3pP/6p1/pp1P4/3B1R2/PPPQ4/1K4R1 b - - 0 25",
            "move_uci": "d8d4",
            "move_san": "Qxd4",
            "player_color": chess.BLACK,
            "cp_loss": 11,
            "best_move": "d8d4",
            "played_best": True,
            "eval_before": 147,
            "eval_after": 136,
            "expected_tactic": "fork",
        },
        {
            "game_id": "game_3",
            "move_number": 74,
            "fen": "5nkq/5R2/2p3pP/6Q1/2B5/pPp5/r7/2K5 b - - 0 37",
            "move_uci": "a2a1",
            "move_san": "Ra1+",
            "player_color": chess.BLACK,
            "cp_loss": 18,
            "best_move": "a2a1",
            "played_best": True,
            "eval_before": -450,
            "eval_after": -468,
            "expected_tactic": "none",
        },
        {
            "game_id": "game_3",
            "move_number": 76,
            "fen": "5nkq/5R2/2p3pP/6Q1/2B5/pPp5/2K5/r7 b - - 2 38",
            "move_uci": "a1a2",
            "move_san": "Ra2+",
            "player_color": chess.BLACK,
            "cp_loss": 21,
            "best_move": "a1a2",
            "played_best": True,
            "eval_before": -465,
            "eval_after": -486,
            "expected_tactic": "none",
        },
        {
            "game_id": "game_3",
            "move_number": 78,
            "fen": "5nkq/5R2/2p3pP/6Q1/2B5/pPpK4/r7/8 b - - 4 39",
            "move_uci": "a2d2",
            "move_san": "Rd2+",
            "player_color": chess.BLACK,
            "cp_loss": 17,
            "best_move": "a2d2",
            "played_best": True,
            "eval_before": -483,
            "eval_after": -500,
            "expected_tactic": "none",
        },
    ]


def test_consecutive_good_moves_classified_independently(consecutive_good_moves_fixture):
    """Assert consecutive GOOD moves in a game are classified independently and do not leak state.
    
    Regression test for bug where detect_fork checked team-wide attacks rather than attacks
    originating from the moved piece, causing consecutive good moves to all falsely inherit 'fork'.
    """
    results_forward = []
    for item in consecutive_good_moves_fixture:
        board = chess.Board(item["fen"])
        move = chess.Move.from_uci(item["move_uci"])
        best = chess.Move.from_uci(item["best_move"]) if item["best_move"] else None
        res = classify_position_taxonomy(
            board=board,
            move=move,
            player_color=item["player_color"],
            eval_before=item["eval_before"],
            eval_after=item["eval_after"],
            cp_loss=item["cp_loss"],
            best_move=best,
            played_best=item["played_best"],
        )
        results_forward.append(res.tactic_type)

    # 1. Assert specific motifs match expectations
    expected_motifs = [item["expected_tactic"] for item in consecutive_good_moves_fixture]
    assert results_forward == expected_motifs

    # 2. Crucially assert that NOT all positions are classified as 'fork'
    assert results_forward != ["fork"] * len(consecutive_good_moves_fixture)
    assert results_forward.count("fork") == 2
    assert results_forward.count("none") == 4
    assert results_forward.count("discovered_attack") == 2

    # 3. Assert order independence (classifying in reverse order yields identical results, zero cross-move state)
    results_reverse = []
    for item in reversed(consecutive_good_moves_fixture):
        board = chess.Board(item["fen"])
        move = chess.Move.from_uci(item["move_uci"])
        best = chess.Move.from_uci(item["best_move"]) if item["best_move"] else None
        res = classify_position_taxonomy(
            board=board,
            move=move,
            player_color=item["player_color"],
            eval_before=item["eval_before"],
            eval_after=item["eval_after"],
            cp_loss=item["cp_loss"],
            best_move=best,
            played_best=item["played_best"],
        )
        results_reverse.append(res.tactic_type)

    assert results_reverse == list(reversed(results_forward))


def test_detect_skewer_scoped_to_moved_piece():
    """Assert skewer detection is scoped to attacks from the moved piece.
    
    Under the old team-wide attack bug, an unrelated Bishop attacking King on e5
    caused an unrelated move (Rc3-c4) to be flagged as a skewer.
    """
    # 1. Unrelated piece attacks king -> move Rc3-c4 must NOT be a skewer
    fen_unrelated = "8/8/5p2/4k3/8/2R5/1B6/4K3 w - - 0 1"
    b_unrelated = chess.Board(fen_unrelated)
    assert detect_skewer(b_unrelated, chess.Move.from_uci("c3c4")) is False

    # 2. Real skewer: Ra1 moves to a5+, checking King on e5 and exposing Queen on h5
    fen_skewer = "8/8/8/4k2q/8/8/8/R3K3 w - - 0 1"
    b_skewer = chess.Board(fen_skewer)
    assert detect_skewer(b_skewer, chess.Move.from_uci("a1a5")) is True


def test_detect_discovered_scoped_to_unmasked_attack():
    """Assert discovered attack requires the attack to be newly unmasked, not pre-existing.
    
    Under the old bug, if a friendly piece already attacked an enemy heavy piece,
    any subsequent move anywhere on the board was tagged as a 'discovered attack'.
    """
    # 1. Pre-existing attack: Rook on e1 already attacks Queen on e8; a3-a4 is NOT discovered
    fen_preexisting = "4q3/8/8/8/8/P7/8/4R1K1 w - - 0 1"
    b_pre = chess.Board(fen_preexisting)
    assert detect_discovered(b_pre, chess.Move.from_uci("a3a4")) is False

    # 2. Genuine discovered attack: Bishop on e2 moves to b5, unmasking Rook e1 onto Queen e8
    fen_real_disc = "4q3/8/8/8/8/8/4B3/4R1K1 w - - 0 1"
    b_real = chess.Board(fen_real_disc)
    assert detect_discovered(b_real, chess.Move.from_uci("e2b5")) is True


def test_detect_overloaded_piece_requires_multiple_attacked_defenders():
    """Assert overloaded piece requires at least 2 defended targets to be actively attacked.
    
    Under the old bug, any piece defending 2 items was considered overloaded if even 1
    of those items was attacked (e.g. King on g8 defending f7 and h7 when only f7 was touched).
    """
    # 1. Single attacked target: King on g8 defends f7 (attacked by Bc4) and h7 (unattacked) -> NOT overloaded
    fen_single_attack = "6k1/5p1p/8/8/2B5/8/8/4K3 w - - 0 1"
    b_single = chess.Board(fen_single_attack)
    assert detect_overloaded_piece(b_single, chess.Move.from_uci("e1e2"), chess.WHITE) is False

    # 2. True overload: King on g8 defends f7 (attacked by Bc4) AND h7 (attacked by Rh5) -> OVERLOADED
    fen_dual_attack = "6k1/5p1p/8/7R/2B5/8/8/4K3 w - - 0 1"
    b_dual = chess.Board(fen_dual_attack)
    assert detect_overloaded_piece(b_dual, chess.Move.from_uci("e1e2"), chess.WHITE) is True


def test_detect_trapped_piece_fixture():
    """Assert trapped piece detection confirms all escape squares are covered."""
    # Bishop on a7 trapped by pawns on b6, c7, attacked by Rook on a8 and Knight on c6
    fen_trapped = "r7/B1p5/1pn5/8/8/8/8/4K2k w - - 0 1"
    b_trapped = chess.Board(fen_trapped)
    is_trap, trap_p = detect_trapped_piece(b_trapped, chess.Move.from_uci("e1e2"), chess.WHITE)
    assert is_trap is True
    assert trap_p == chess.BISHOP

    # Non-trapped piece: Bishop has safe retreat along open diagonal (b6, c5)
    fen_safe = "8/B1p5/2n5/8/8/8/8/4K2k w - - 0 1"
    b_safe = chess.Board(fen_safe)
    is_trap_safe, _ = detect_trapped_piece(b_safe, chess.Move.from_uci("e1e2"), chess.WHITE)
    assert is_trap_safe is False


def test_detect_hanging_piece_fixture():
    """Assert hanging piece detects undefended pieces and ignores defended pieces."""
    # 1. Knight on e4 attacked by Re8 and undefended -> hanging
    fen_hanging = "4r1k1/8/8/8/4N3/8/8/4K3 w - - 0 1"
    b_hanging = chess.Board(fen_hanging)
    is_hang, hung_p = detect_hanging(b_hanging, chess.Move.from_uci("e1d1"), chess.WHITE)
    assert is_hang is True
    assert hung_p == chess.KNIGHT

    # 2. Knight on e4 defended by pawn on d3 -> NOT hanging
    fen_defended = "4r1k1/8/8/8/4N3/3P4/8/4K3 w - - 0 1"
    b_defended = chess.Board(fen_defended)
    is_hang_def, _ = detect_hanging(b_defended, chess.Move.from_uci("e1d1"), chess.WHITE)
    assert is_hang_def is False


def test_detect_sacrifice_genuine_vs_favorable_capture():
    """Assert sacrifice requires giving up material on a defended square, rejecting favorable/free captures.
    
    Under the old bug, Queen takes undefended Rook or Pawn was called a sacrifice simply because
    attacker value (9) > captured value (1 or 5) + 1.
    """
    # 1. Buggy favorable capture: Queen captures undefended Rook on d8 -> NOT a sacrifice
    fen_free_rook = "3r2k1/5ppp/8/8/8/8/8/3Q2K1 w - - 0 1"
    b_free_rook = chess.Board(fen_free_rook)
    assert detect_sacrifice_missed(b_free_rook, chess.Move.from_uci("g1f1"), chess.Move.from_uci("d1d8"), cp_loss=300) is False

    # 2. Buggy favorable capture: Queen captures undefended Pawn on d6 -> NOT a sacrifice
    fen_free_pawn = "3k4/8/3p4/8/8/8/8/3Q2K1 w - - 0 1"
    b_free_pawn = chess.Board(fen_free_pawn)
    assert detect_sacrifice_missed(b_free_pawn, chess.Move.from_uci("g1f1"), chess.Move.from_uci("d1d6"), cp_loss=300) is False

    # 3. Genuine Queen sacrifice: Queen captures Rook on d8 defended by King on c7 (Kc7xd8)
    fen_queen_sac = "2kr4/ppp5/8/8/8/8/8/3Q2K1 w - - 0 1"
    b_queen_sac = chess.Board(fen_queen_sac)
    best_move_qs = chess.Move.from_uci("d1d8")
    assert detect_sacrifice_missed(b_queen_sac, chess.Move.from_uci("g1f1"), best_move_qs, cp_loss=300) is True

    # 4. Genuine Exchange sacrifice: Rook on e1 captures Bishop on e5 defended by pawn d6 (d6xe5)
    fen_exch_sac = "4k3/8/3p4/4b3/8/8/8/4R1K1 w - - 0 1"
    b_exch_sac = chess.Board(fen_exch_sac)
    best_move_es = chess.Move.from_uci("e1e5")
    assert detect_sacrifice_missed(b_exch_sac, chess.Move.from_uci("g1f1"), best_move_es, cp_loss=300) is True


def test_detect_zwischenzug_fixture():
    """Assert zwischenzug detects in-between check when played move was quiet."""
    fen_zw = "3r2k1/5ppp/8/8/8/8/8/3Q2K1 w - - 0 1"
    b_zw = chess.Board(fen_zw)
    best_move_zw = chess.Move.from_uci("d1d8")
    assert detect_zwischenzug(b_zw, chess.Move.from_uci("g1f1"), best_move_zw) is True



def test_detect_pin_scoped_to_moved_piece():
    """Assert pin detection requires the moved piece to create the pin, ignoring pre-existing pins."""
    # 1. Pre-existing pin: White Bishop on a4 pins Black Knight on d7 to King on e8.
    # Unrelated move e1e2 must NOT be classified as pinning.
    fen_existing_pin = "4k3/3n4/8/B7/8/8/8/4K3 w - - 0 1"
    b_existing = chess.Board(fen_existing_pin)
    assert detect_pin(b_existing, chess.Move.from_uci("e1e2")) is False

    # 2. Real pin: White Bishop on c1 moves to a4, creating the pin on Nd7 against Ke8.
    fen_pin = "4k3/3n4/8/8/8/8/8/2B1K3 w - - 0 1"
    b_pin = chess.Board(fen_pin)
    assert detect_pin(b_pin, chess.Move.from_uci("c1a4")) is True


def test_detect_back_rank_scoped_to_moved_piece():
    """Assert back-rank check/mate requires the moved rook/queen to actually attack the king."""
    # 1. Rook moves to back rank, but enemy piece blocks it from king (check delivered by discovered bishop)
    # White: King g1, Bishop b2, Rook a1. Black: King g8, pawns f7/g7/h7, Rook e8.
    # If a rook moves to a8 while e8 blocks a8 from g8, it does NOT directly attack g8.
    fen_blocked = "4r1k1/5ppp/8/8/8/8/1B6/R5K1 w - - 0 1"
    b_blocked = chess.Board(fen_blocked)
    # Move Bc3+ gives check, but Ra1a8 is not a check; even if check was already present,
    # moving Ra1-a8 when blocked by e8 is not back-rank check
    assert detect_back_rank(b_blocked, chess.Move.from_uci("a1a8")) is False

    # 2. Genuine back-rank mate: Ra1 moves to a8#, directly checking/mating King on g8
    fen_br = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
    b_br = chess.Board(fen_br)
    assert detect_back_rank(b_br, chess.Move.from_uci("a1a8")) is True


