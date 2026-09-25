"""Tactical blunder taxonomy classification for chess positions.

Detects 15+ tactical motifs adapted from chessIQ phase4_taxonomy.py.
All functions are pure, deterministic, and independent of any UI framework.
"""

from typing import Optional, Tuple, Dict, Any, List
import chess
from chess_commentator.analysis.constants import (
    PIECE_VALUE,
    PIECE_NAMES,
    CP_CAP,
    GOOD_THRESHOLD,
    INACCURACY_THRESHOLD,
    MISTAKE_THRESHOLD,
)
from chess_commentator.analysis.models import TaxonomyResult, BoardFeatures
from chess_commentator.analysis.features import get_game_phase


# ── Individual Tactical Detectors ────────────────────────────────────────────

def detect_fork(board: chess.Board, move: chess.Move) -> bool:
    """Detect if move results in a single piece attacking two or more valuable pieces."""
    b = board.copy()
    b.push(move)
    piece = b.piece_at(move.to_square)
    if not piece:
        return False
    attacked = 0
    valuable = (chess.QUEEN, chess.ROOK, chess.KING, chess.BISHOP, chess.KNIGHT)
    for sq in b.attacks(move.to_square):
        target = b.piece_at(sq)
        if target and target.color != piece.color and target.piece_type in valuable:
            attacked += 1
    return attacked >= 2


def extract_fork_elements(board: chess.Board, move: chess.Move) -> Optional[Dict[str, Any]]:
    """Extract constituent forking and target pieces/squares."""
    b = board.copy()
    b.push(move)
    piece = b.piece_at(move.to_square)
    if not piece:
        return None
    valuable = (chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)
    targets = []
    for sq in b.attacks(move.to_square):
        target = b.piece_at(sq)
        if target and target.color != piece.color and target.piece_type in valuable:
            targets.append((sq, target))

    # Sort targets by piece value descending
    targets.sort(
        key=lambda x: PIECE_VALUE.get(x[1].piece_type, 100 if x[1].piece_type == chess.KING else 0),
        reverse=True,
    )
    if len(targets) >= 2:
        forking_piece = f"{chess.piece_name(piece.piece_type).title()} on {chess.square_name(move.to_square)}"
        target_pieces = [
            f"{chess.piece_name(t.piece_type).title()} on {chess.square_name(sq)}"
            for sq, t in targets
        ]
        return {
            "forking_piece": forking_piece,
            "target_pieces": target_pieces,
        }
    return None


def detect_pin(board: chess.Board, move: chess.Move) -> bool:
    """Detect if move pins an opponent piece along a line to their king."""
    b = board.copy()
    b.push(move)
    piece = b.piece_at(move.to_square)
    if not piece or piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        return False
    opp = b.turn
    ksq = b.king(opp)
    if ksq is None:
        return False

    ray_squares = list(chess.SquareSet.between(move.to_square, ksq))
    if not ray_squares:
        return False

    if not b.is_attacked_by(piece.color, ksq):
        opp_pieces_between = []
        for sq in ray_squares:
            p = b.piece_at(sq)
            if p and p.color == opp:
                opp_pieces_between.append((sq, p))
            elif p and p.color == piece.color:
                return False  # Friendly piece blocks the ray

        if len(opp_pieces_between) == 1:
            pin_sq, pin_piece = opp_pieces_between[0]
            b.remove_piece_at(pin_sq)
            is_pinned = ksq in b.attacks(move.to_square)
            b.set_piece_at(pin_sq, pin_piece)
            return is_pinned

    return False


def extract_pin_elements(board: chess.Board, move: chess.Move) -> Optional[Dict[str, Any]]:
    """Extract constituent pinning piece, pinned piece, and shielded piece/king."""
    b = board.copy()
    b.push(move)
    piece = b.piece_at(move.to_square)
    if not piece or piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        return None
    opp = b.turn

    # 1. Absolute pin to King
    ksq = b.king(opp)
    if ksq is not None:
        ray_squares = list(chess.SquareSet.between(move.to_square, ksq))
        if ray_squares:
            opp_pieces = [
                (sq, b.piece_at(sq))
                for sq in ray_squares
                if b.piece_at(sq) and b.piece_at(sq).color == opp
            ]
            friendly_pieces = [
                sq
                for sq in ray_squares
                if b.piece_at(sq) and b.piece_at(sq).color == piece.color
            ]
            if len(opp_pieces) == 1 and not friendly_pieces:
                psq, pp = opp_pieces[0]
                return {
                    "pinning_piece": f"{chess.piece_name(piece.piece_type).title()} on {chess.square_name(move.to_square)}",
                    "pinned_piece": f"{chess.piece_name(pp.piece_type).title()} on {chess.square_name(psq)}",
                    "shielded_piece": f"King on {chess.square_name(ksq)}",
                }

    # 2. Relative pin to Queen
    for qsq in b.pieces(chess.QUEEN, opp):
        ray_squares = list(chess.SquareSet.between(move.to_square, qsq))
        if ray_squares:
            opp_pieces = [
                (sq, b.piece_at(sq))
                for sq in ray_squares
                if b.piece_at(sq) and b.piece_at(sq).color == opp
            ]
            friendly_pieces = [
                sq
                for sq in ray_squares
                if b.piece_at(sq) and b.piece_at(sq).color == piece.color
            ]
            if len(opp_pieces) == 1 and not friendly_pieces:
                psq, pp = opp_pieces[0]
                return {
                    "pinning_piece": f"{chess.piece_name(piece.piece_type).title()} on {chess.square_name(move.to_square)}",
                    "pinned_piece": f"{chess.piece_name(pp.piece_type).title()} on {chess.square_name(psq)}",
                    "shielded_piece": f"Queen on {chess.square_name(qsq)}",
                }

    # 3. Relative pin to Rook
    for rsq in b.pieces(chess.ROOK, opp):
        ray_squares = list(chess.SquareSet.between(move.to_square, rsq))
        if ray_squares:
            opp_pieces = [
                (sq, b.piece_at(sq))
                for sq in ray_squares
                if b.piece_at(sq) and b.piece_at(sq).color == opp
            ]
            friendly_pieces = [
                sq
                for sq in ray_squares
                if b.piece_at(sq) and b.piece_at(sq).color == piece.color
            ]
            if len(opp_pieces) == 1 and not friendly_pieces:
                psq, pp = opp_pieces[0]
                return {
                    "pinning_piece": f"{chess.piece_name(piece.piece_type).title()} on {chess.square_name(move.to_square)}",
                    "pinned_piece": f"{chess.piece_name(pp.piece_type).title()} on {chess.square_name(psq)}",
                    "shielded_piece": f"Rook on {chess.square_name(rsq)}",
                }

    return None


def detect_skewer(board: chess.Board, move: chess.Move) -> bool:
    """Detect skewer: attacking high-value piece with another enemy piece behind it."""
    b = board.copy()
    b.push(move)
    piece = b.piece_at(move.to_square)
    if not piece or piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        return False
    opp = not piece.color

    for sq in b.attacks(move.to_square):
        target = b.piece_at(sq)
        if target and target.color == opp and target.piece_type in (chess.KING, chess.QUEEN):
            df_file = chess.square_file(sq) - chess.square_file(move.to_square)
            df_rank = chess.square_rank(sq) - chess.square_rank(move.to_square)
            d_file = (1 if df_file > 0 else -1) if df_file != 0 else 0
            d_rank = (1 if df_rank > 0 else -1) if df_rank != 0 else 0

            cur_file = chess.square_file(sq) + d_file
            cur_rank = chess.square_rank(sq) + d_rank
            while 0 <= cur_file < 8 and 0 <= cur_rank < 8:
                behind_sq = chess.square(cur_file, cur_rank)
                behind_piece = b.piece_at(behind_sq)
                if behind_piece:
                    if behind_piece.color == opp:
                        return True
                    break
                cur_file += d_file
                cur_rank += d_rank
    return False


def extract_skewer_elements(board: chess.Board, move: chess.Move) -> Optional[Dict[str, Any]]:
    """Extract attacking piece, front piece, and back piece of a skewer."""
    b = board.copy()
    b.push(move)
    piece = b.piece_at(move.to_square)
    if not piece or piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        return None
    opp = not piece.color

    for sq in b.attacks(move.to_square):
        target = b.piece_at(sq)
        if target and target.color == opp and target.piece_type in (chess.KING, chess.QUEEN, chess.ROOK):
            df_file = chess.square_file(sq) - chess.square_file(move.to_square)
            df_rank = chess.square_rank(sq) - chess.square_rank(move.to_square)
            d_file = (1 if df_file > 0 else -1) if df_file != 0 else 0
            d_rank = (1 if df_rank > 0 else -1) if df_rank != 0 else 0

            cur_file = chess.square_file(sq) + d_file
            cur_rank = chess.square_rank(sq) + d_rank
            while 0 <= cur_file < 8 and 0 <= cur_rank < 8:
                behind_sq = chess.square(cur_file, cur_rank)
                behind_piece = b.piece_at(behind_sq)
                if behind_piece:
                    if behind_piece.color == opp:
                        return {
                            "attacking_piece": f"{chess.piece_name(piece.piece_type).title()} on {chess.square_name(move.to_square)}",
                            "front_piece": f"{chess.piece_name(target.piece_type).title()} on {chess.square_name(sq)}",
                            "back_piece": f"{chess.piece_name(behind_piece.piece_type).title()} on {chess.square_name(behind_sq)}",
                        }
                    break
                cur_file += d_file
                cur_rank += d_rank
    return None


def detect_back_rank(board: chess.Board, move: chess.Move) -> bool:
    """Detect back-rank check/mate: rook or queen attacking enemy king on rank 0 or 7."""
    piece = board.piece_at(move.from_square)
    if not piece or piece.piece_type not in (chess.ROOK, chess.QUEEN):
        return False
    b = board.copy()
    b.push(move)
    if not b.is_check():
        return False
    ksq = b.king(b.turn)
    if ksq is None:
        return False
    back_rank = chess.square_rank(ksq)
    if back_rank not in (0, 7):
        return False
    return chess.square_rank(move.to_square) == back_rank and (ksq in b.attacks(move.to_square))


def detect_discovered(board: chess.Board, move: chess.Move) -> bool:
    """Detect discovered attack: moving a piece unmasks an attack from another piece."""
    piece = board.piece_at(move.from_square)
    if not piece:
        return False
    b = board.copy()
    b.push(move)
    for sq in chess.SQUARES:
        target = b.piece_at(sq)
        if target and target.color != piece.color:
            if target.piece_type in (chess.QUEEN, chess.ROOK, chess.KING):
                for att_sq in b.attackers(piece.color, sq):
                    if att_sq != move.to_square and att_sq not in board.attackers(piece.color, sq):
                        return True
    return False


def extract_discovered_elements(board: chess.Board, move: chess.Move) -> Optional[Dict[str, Any]]:
    """Extract moved piece, revealed attacker, and target piece of discovered attack."""
    piece = board.piece_at(move.from_square)
    if not piece:
        return None
    b = board.copy()
    b.push(move)
    for sq in chess.SQUARES:
        target = b.piece_at(sq)
        if target and target.color != piece.color and target.piece_type in (chess.QUEEN, chess.ROOK, chess.KING, chess.BISHOP, chess.KNIGHT):
            for att_sq in b.attackers(piece.color, sq):
                if att_sq != move.to_square and att_sq not in board.attackers(piece.color, sq):
                    rev = b.piece_at(att_sq)
                    if rev:
                        return {
                            "moved_piece": f"{chess.piece_name(piece.piece_type).title()} to {chess.square_name(move.to_square)}",
                            "revealed_attacker": f"{chess.piece_name(rev.piece_type).title()} on {chess.square_name(att_sq)}",
                            "target_piece": f"{chess.piece_name(target.piece_type).title()} on {chess.square_name(sq)}",
                        }
    return None


def detect_hanging(
    board: chess.Board, move: chess.Move, color: chess.Color
) -> Tuple[bool, Optional[chess.PieceType]]:
    """Detect if a major/minor piece was left hanging (attacked and undefended)."""
    b = board.copy()
    b.push(move)
    opp = not color
    for sq in chess.SQUARES:
        piece = b.piece_at(sq)
        if piece and piece.color == color:
            if piece.piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                if b.attackers(opp, sq) and not b.attackers(color, sq):
                    return True, piece.piece_type
    return False, None


def detect_trapped_piece(
    board: chess.Board, move: chess.Move, color: chess.Color
) -> Tuple[bool, Optional[chess.PieceType]]:
    """Detect if player's piece has no safe squares after the move."""
    b = board.copy()
    b.push(move)
    opp = not color
    for sq in chess.SQUARES:
        piece = b.piece_at(sq)
        if piece and piece.color == color:
            if piece.piece_type in (chess.BISHOP, chess.KNIGHT, chess.ROOK, chess.QUEEN):
                if b.attackers(opp, sq):
                    safe_squares = 0
                    b2 = b.copy()
                    b2.turn = color
                    for escape in b2.generate_legal_moves():
                        if escape.from_square == sq:
                            b3 = b2.copy()
                            b3.push(escape)
                            if not b3.attackers(opp, escape.to_square):
                                safe_squares += 1
                                break
                    if safe_squares == 0:
                        return True, piece.piece_type
    return False, None


def extract_trapped_elements(
    board: chess.Board, move: chess.Move, color: chess.Color
) -> Optional[Dict[str, Any]]:
    """Extract trapped piece and constraining attackers/defenders."""
    b = board.copy()
    b.push(move)
    opp = not color
    for sq in chess.SQUARES:
        piece = b.piece_at(sq)
        if piece and piece.color == color:
            if piece.piece_type in (chess.BISHOP, chess.KNIGHT, chess.ROOK, chess.QUEEN):
                opp_atts = list(b.attackers(opp, sq))
                if opp_atts:
                    safe_squares = 0
                    b2 = b.copy()
                    b2.turn = color
                    for escape in b2.generate_legal_moves():
                        if escape.from_square == sq:
                            b3 = b2.copy()
                            b3.push(escape)
                            if not b3.attackers(opp, escape.to_square):
                                safe_squares += 1
                                break
                    if safe_squares == 0:
                        constraining = [
                            f"{chess.piece_name(b.piece_at(a).piece_type).title()} on {chess.square_name(a)}"
                            for a in opp_atts
                            if b.piece_at(a)
                        ]
                        return {
                            "trapped_piece": f"{chess.piece_name(piece.piece_type).title()} on {chess.square_name(sq)}",
                            "constraining_pieces": constraining,
                        }
    return None


def detect_weak_back_rank(board: chess.Board, move: chess.Move, color: chess.Color) -> bool:
    """Detect if the player's own back rank becomes dangerously vulnerable."""
    b = board.copy()
    b.push(move)
    opp = not color
    back_rank = 0 if color == chess.WHITE else 7
    king_sq = b.king(color)
    if not king_sq or chess.square_rank(king_sq) != back_rank:
        return False
    for sq in b.pieces(chess.ROOK, opp) | b.pieces(chess.QUEEN, opp):
        if b.is_attacked_by(color, sq):
            continue
        if chess.square_rank(sq) == back_rank:
            return True
    return False


def detect_pawn_fork_missed(board: chess.Board, move: chess.Move) -> bool:
    """Detect missed pawn fork opportunity available to player."""
    color = board.turn
    opp = not color
    for sq in board.pieces(chess.PAWN, color):
        pawn_move = chess.Move(sq, sq + (8 if color == chess.WHITE else -8))
        if pawn_move in board.legal_moves and pawn_move != move:
            b2 = board.copy()
            b2.push(pawn_move)
            attacked = sum(
                1 for s in b2.attacks(pawn_move.to_square)
                if b2.piece_at(s) and b2.piece_at(s).color == opp
                and b2.piece_at(s).piece_type in (chess.QUEEN, chess.ROOK, chess.KNIGHT, chess.BISHOP)
            )
            if attacked >= 2:
                return True
    return False


def detect_overloaded_piece(board: chess.Board, move: chess.Move, color: chess.Color) -> bool:
    """Detect if opponent piece is overloaded (defending multiple attacked items)."""
    b = board.copy()
    b.push(move)
    opp = not color
    for sq in chess.SQUARES:
        piece = b.piece_at(sq)
        if piece and piece.color == opp:
            defended = []
            for sq2 in chess.SQUARES:
                t = b.piece_at(sq2)
                if t and t.color == opp and sq2 != sq:
                    if b.is_attacked_by(opp, sq2) and sq in b.attackers(opp, sq2):
                        defended.append(sq2)
            if len(defended) >= 2:
                attacked_defended = [def_sq for def_sq in defended if b.attackers(color, def_sq)]
                if len(attacked_defended) >= 2:
                    return True
    return False


def detect_zwischenzug(
    board: chess.Board, move: chess.Move, best_move: Optional[chess.Move]
) -> bool:
    """Detect missed zwischenzug (e.g. an intermediate check or threat instead of routine recapture)."""
    if not best_move or best_move not in board.legal_moves:
        return False
    b = board.copy()
    b.push(best_move)
    played_capture = board.is_capture(move)
    best_gives_check = b.is_check()
    return best_gives_check and not played_capture


def detect_sacrifice_missed(
    board: chess.Board, move: chess.Move, best_move: Optional[chess.Move], cp_loss: int
) -> bool:
    """Detect missed tactical sacrifice.
    
    A sacrifice occurs when the engine's best move intentionally gives up material:
    1. The destination square is defended/attacked by the opponent in the resulting position.
    2. The move either:
       - Captures a piece of lower value than the attacking piece (att_val > cap_val, e.g. RxB, QxR, BxP on defended square).
       - Captures an equal value piece (att_val == cap_val) where the opponent can recapture with a lower-value piece (e.g., RxR defended by pawn).
       - Is a non-capture piece move (att_val >= 3) to a square attacked by the opponent.
    """
    if not best_move or cp_loss < 150 or best_move not in board.legal_moves:
        return False

    attacker = board.piece_at(best_move.from_square)
    if not attacker:
        return False
    att_val = PIECE_VALUE.get(attacker.piece_type, 0)

    b_after = board.copy()
    b_after.push(best_move)
    opp = not board.turn

    # The destination square must be attacked/defended by the opponent
    if not b_after.is_attacked_by(opp, best_move.to_square):
        return False

    opp_attackers = list(b_after.attackers(opp, best_move.to_square))
    lowest_opp_att_val = min(
        [PIECE_VALUE.get(b_after.piece_at(sq).piece_type, 0) for sq in opp_attackers if b_after.piece_at(sq)],
        default=999,
    )

    if board.is_capture(best_move):
        captured = board.piece_at(best_move.to_square)
        cap_val = PIECE_VALUE.get(captured.piece_type, 0) if captured else 0

        # Attacking piece is more valuable than captured piece on defended square
        if att_val > cap_val:
            return True
        # Equal exchange where opponent can recapture with a lower-value piece
        if att_val == cap_val and lowest_opp_att_val < att_val:
            return True
    else:
        # Non-capture piece sacrifice (Knight, Bishop, Rook, Queen) into an attacked square
        if att_val >= 3:
            friendly_defenders = list(b_after.attackers(board.turn, best_move.to_square))
            if not friendly_defenders or lowest_opp_att_val < att_val:
                return True

    return False



# ── Master Taxonomy Classifier ───────────────────────────────────────────────

def classify_position_taxonomy(
    board: chess.Board,
    move: chess.Move,
    player_color: chess.Color,
    eval_before: Optional[int],
    eval_after: Optional[int],
    cp_loss: Optional[int],
    best_move: Optional[chess.Move] = None,
    played_best: bool = False,
    mode: str = "deep",
    features: Optional[BoardFeatures] = None,
    move_number: int = 20,
) -> TaxonomyResult:
    """Classify the move into a tactical taxonomy category."""
    loss = cp_loss or 0

    # 0. Check if played move delivers checkmate
    b_after = board.copy()
    b_after.push(move)
    if b_after.is_checkmate():
        if detect_back_rank(board, move):
            return TaxonomyResult(
                mistake_category="none",
                tactic_type="back_rank_mate",
                mate_missed=0,
            )
        return TaxonomyResult(
            mistake_category="none",
            tactic_type="checkmate",
            mate_missed=0,
        )

    # Clean or good moves: detect executed positive tactical motifs
    if played_best or loss < GOOD_THRESHOLD:
        if detect_fork(board, move):
            piece = board.piece_at(move.from_square)
            fork_type = (
                "knight_fork" if piece and piece.piece_type == chess.KNIGHT
                else "pawn_fork" if piece and piece.piece_type == chess.PAWN
                else "fork"
            )
            return TaxonomyResult(
                mistake_category="none",
                tactic_type=fork_type,
                tactical_elements=extract_fork_elements(board, move),
            )
        if detect_pin(board, move):
            return TaxonomyResult(
                mistake_category="none",
                tactic_type="pin",
                tactical_elements=extract_pin_elements(board, move),
            )
        if detect_skewer(board, move):
            return TaxonomyResult(
                mistake_category="none",
                tactic_type="skewer",
                tactical_elements=extract_skewer_elements(board, move),
            )
        if detect_discovered(board, move):
            return TaxonomyResult(
                mistake_category="none",
                tactic_type="discovered_attack",
                tactical_elements=extract_discovered_elements(board, move),
            )
        return TaxonomyResult(mistake_category="none", tactic_type="none")

    # 1. Missed forced mate (player had forced mate before moving)
    if eval_before is not None and abs(eval_before) >= CP_CAP and loss >= 200:
        return TaxonomyResult(
            mistake_category="missed_mate",
            tactic_type="missed_mate",
            mate_missed=1,
        )

    # 2. Hanging piece (player left a piece completely undefended)
    is_hang, hung_piece = detect_hanging(board, move, player_color)
    if is_hang and loss >= 150 and hung_piece:
        piece_name = PIECE_NAMES.get(hung_piece, "piece")
        return TaxonomyResult(
            mistake_category="hanging_piece",
            tactic_type="hanging_piece",
            piece_lost=piece_name,
        )

    # 3. Trapped piece
    if mode == "deep":
        is_trap, trap_piece = detect_trapped_piece(board, move, player_color)
        if is_trap and loss >= 150 and trap_piece:
            piece_name = PIECE_NAMES.get(trap_piece, "piece")
            return TaxonomyResult(
                mistake_category="trapped_piece",
                tactic_type=f"trapped_{piece_name}",
                piece_lost=piece_name,
                tactical_elements=extract_trapped_elements(board, move, player_color),
            )

    # 4. Missed tactic on best move
    if best_move and loss >= 150 and not played_best:
        if detect_back_rank(board, best_move):
            return TaxonomyResult(mistake_category="missed_tactic", tactic_type="back_rank_mate")

        if detect_fork(board, best_move):
            piece = board.piece_at(best_move.from_square)
            fork_type = (
                "knight_fork" if piece and piece.piece_type == chess.KNIGHT
                else "pawn_fork" if piece and piece.piece_type == chess.PAWN
                else "fork"
            )
            return TaxonomyResult(
                mistake_category="missed_tactic",
                tactic_type=fork_type,
                tactical_elements=extract_fork_elements(board, best_move),
            )

        if detect_pin(board, best_move):
            return TaxonomyResult(
                mistake_category="missed_tactic",
                tactic_type="pin",
                tactical_elements=extract_pin_elements(board, best_move),
            )

        if detect_skewer(board, best_move):
            return TaxonomyResult(
                mistake_category="missed_tactic",
                tactic_type="skewer",
                tactical_elements=extract_skewer_elements(board, best_move),
            )

        if detect_discovered(board, best_move):
            return TaxonomyResult(
                mistake_category="missed_tactic",
                tactic_type="discovered_attack",
                tactical_elements=extract_discovered_elements(board, best_move),
            )

        if mode == "deep":
            if detect_zwischenzug(board, move, best_move):
                return TaxonomyResult(mistake_category="missed_tactic", tactic_type="zwischenzug")
            if detect_sacrifice_missed(board, move, best_move, loss):
                return TaxonomyResult(mistake_category="missed_tactic", tactic_type="missed_sacrifice")
            if detect_overloaded_piece(board, move, player_color):
                return TaxonomyResult(mistake_category="missed_tactic", tactic_type="overloaded_piece")

        # Phase-based calculation error
        phase = features.phase if features else get_game_phase(board, move_number)
        calc_type = {
            "opening": "opening_calculation",
            "middlegame": "middlegame_calculation",
            "endgame": "endgame_calculation",
        }.get(phase, "calculation_error")
        return TaxonomyResult(mistake_category="missed_tactic", tactic_type=calc_type)

    # 5. Time pressure blunder
    if features and features.time_pressure and loss >= MISTAKE_THRESHOLD:
        return TaxonomyResult(mistake_category="time_pressure_blunder", tactic_type="time_pressure")

    # 6. King safety error
    if features and not features.castled and features.open_files_near_king >= 2 and loss >= INACCURACY_THRESHOLD:
        return TaxonomyResult(mistake_category="king_safety_error", tactic_type="king_exposed")

    if mode == "deep" and detect_weak_back_rank(board, move, player_color) and loss >= INACCURACY_THRESHOLD:
        return TaxonomyResult(mistake_category="king_safety_error", tactic_type="weak_back_rank")

    # 7. Phase errors
    phase = features.phase if features else get_game_phase(board, move_number)
    if phase == "opening" and loss >= INACCURACY_THRESHOLD:
        return TaxonomyResult(mistake_category="opening_error", tactic_type="opening_principle")

    if phase == "endgame" and loss >= INACCURACY_THRESHOLD:
        has_pawns = bool(board.pieces(chess.PAWN, player_color))
        has_rooks = bool(board.pieces(chess.ROOK, player_color))
        eg_type = "rook_endgame" if has_rooks else "pawn_endgame" if has_pawns else "endgame_technique"
        return TaxonomyResult(mistake_category="endgame_error", tactic_type=eg_type)

    if features and (features.isolated_pawns >= 2 or features.doubled_pawns >= 2) and loss >= INACCURACY_THRESHOLD:
        return TaxonomyResult(mistake_category="pawn_structure_error", tactic_type="weak_pawns")

    return TaxonomyResult(
        mistake_category="blunder_other" if loss >= MISTAKE_THRESHOLD else "mistake_other",
        tactic_type="blunder_other" if loss >= MISTAKE_THRESHOLD else "mistake_other",
    )
