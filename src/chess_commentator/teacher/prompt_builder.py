from typing import Dict, Any, Optional
import chess

import atexit

from chess_commentator.analysis.models import PositionAnalysis
from chess_commentator.analysis.engine import StockfishEngine
from chess_commentator.analysis.taxonomy import (
    extract_fork_elements,
    extract_pin_elements,
    extract_skewer_elements,
    extract_discovered_elements,
    extract_trapped_elements,
)

_SF_ENGINE_INSTANCE: Optional[StockfishEngine] = None


CONTINUATION_TACTIC_TYPES = {
    "missed_sacrifice",
    "zwischenzug",
    "missed_mate",
    "overloaded_piece",
    "hanging_piece",
    "checkmate",
}

VERIFICATION_TACTIC_TYPES = {
    "fork",
    "knight_fork",
    "pawn_fork",
    "pin",
    "skewer",
    "discovered_attack",
    "trapped_bishop",
    "trapped_knight",
    "trapped_rook",
    "trapped_queen",
    "trapped_piece",
}


def compute_tactical_verification(
    fen: str,
    move_uci: str,
    tactic: str = "",
    player_color: str = "white",
    best_move_uci: Optional[str] = None,
    mistake_category: str = "none",
    tactical_elements: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Compute the verified tactical elements block for executed or missed tactics."""
    if not tactical_elements:
        try:
            b = chess.Board(fen)
            target_uci = (
                best_move_uci
                if (mistake_category == "missed_tactic" and best_move_uci)
                else move_uci
            )
            if not target_uci:
                return None
            m = chess.Move.from_uci(target_uci)
            if m not in b.legal_moves:
                return None

            color = chess.WHITE if player_color.lower() == "white" else chess.BLACK
            if tactic in ("fork", "knight_fork", "pawn_fork"):
                tactical_elements = extract_fork_elements(b, m)
            elif tactic == "pin":
                tactical_elements = extract_pin_elements(b, m)
            elif tactic == "skewer":
                tactical_elements = extract_skewer_elements(b, m)
            elif tactic == "discovered_attack":
                tactical_elements = extract_discovered_elements(b, m)
            elif tactic.startswith("trapped_"):
                tactical_elements = extract_trapped_elements(b, m, color)
        except Exception:
            return None

    if not tactical_elements:
        return None

    lines = ["Tactical Elements:"]
    # Fork
    if "forking_piece" in tactical_elements:
        lines.append(f"- Forking Piece: {tactical_elements['forking_piece']}")
    if "target_pieces" in tactical_elements:
        lines.append(f"- Target Pieces: {', '.join(tactical_elements['target_pieces'])}")
    # Pin
    if "pinning_piece" in tactical_elements:
        lines.append(f"- Pinning Piece: {tactical_elements['pinning_piece']}")
    if "pinned_piece" in tactical_elements:
        lines.append(f"- Pinned Piece: {tactical_elements['pinned_piece']}")
    if "shielded_piece" in tactical_elements:
        lines.append(f"- Shielded Piece: {tactical_elements['shielded_piece']}")
    # Skewer
    if "attacking_piece" in tactical_elements:
        lines.append(f"- Attacking Piece: {tactical_elements['attacking_piece']}")
    if "front_piece" in tactical_elements:
        lines.append(f"- Front Piece: {tactical_elements['front_piece']}")
    if "back_piece" in tactical_elements:
        lines.append(f"- Back Piece: {tactical_elements['back_piece']}")
    # Discovered attack
    if "moved_piece" in tactical_elements:
        lines.append(f"- Moved Piece: {tactical_elements['moved_piece']}")
    if "revealed_attacker" in tactical_elements:
        lines.append(f"- Revealed Attacker: {tactical_elements['revealed_attacker']}")
    if "target_piece" in tactical_elements:
        lines.append(f"- Target Piece: {tactical_elements['target_piece']}")
    # Trapped piece
    if "trapped_piece" in tactical_elements:
        lines.append(f"- Trapped Piece: {tactical_elements['trapped_piece']}")
    if "constraining_pieces" in tactical_elements:
        lines.append(f"- Constraining Pieces: {', '.join(tactical_elements['constraining_pieces'])}")

    lines.append(
        "Base your explanation only on the position, the move played, and the tactical elements shown above — "
        "don't introduce additional moves, checks, or captures beyond what's given here or already visible on the board."
    )
    return "\n".join(lines)


def compute_tactical_continuation(
    fen: str,
    best_move_uci: Optional[str],
    tactic: str = "",
    engine: Optional[StockfishEngine] = None,
) -> Optional[str]:
    """Compute the verified tactical continuation (single ply+1 or multi-ply mating PV)."""
    if not best_move_uci:
        return None
    try:
        b = chess.Board(fen)
        bm = chess.Move.from_uci(best_move_uci)
        if bm not in b.legal_moves:
            return None
        b_after = b.copy()
        b_after.push(bm)
        if b_after.is_checkmate():
            return "None (delivers immediate checkmate)"
        if b_after.legal_moves.count() == 0:
            return "None (game ends)"

        try:
            limit = chess.engine.Limit(depth=10)
            if engine is not None:
                info = engine._engine.analyse(b_after, limit)
                pv = info.get("pv", [])
            else:
                with StockfishEngine(default_depth=10) as sf:
                    info = sf._engine.analyse(b_after, limit)
                    pv = info.get("pv", [])
            if pv:
                    if tactic in ("missed_mate", "checkmate"):
                        # Short forced line aligned with checker tree (up to 2 plies from b_after: plies 2-3 overall)
                        b_pv = b_after.copy()
                        pv_sans = []
                        for m in pv[:2]:
                            if m in b_pv.legal_moves:
                                pv_sans.append(b_pv.san(m))
                                b_pv.push(m)
                                if b_pv.is_checkmate():
                                    break
                        if pv_sans:
                            return " ".join(pv_sans)
                    # Standard single ply+1 reply
                    if pv[0] in b_after.legal_moves:
                        return b_after.san(pv[0])
        except Exception:
            pass

        # Fallback to recapture or first legal move
        recaps = [m for m in b_after.legal_moves if m.to_square == bm.to_square]
        if recaps:
            return b_after.san(recaps[0])
        return b_after.san(next(iter(b_after.legal_moves)))
    except Exception:
        return None


def compute_opponent_response(fen: str, best_move_uci: Optional[str]) -> Optional[str]:
    """Compute the single legal opponent response (ply+1) to best_move from the board state."""
    return compute_tactical_continuation(fen, best_move_uci, tactic="standard")




TEACHER_SYSTEM_PROMPT = """You are an elite chess grandmaster commentator and coach.
Your job is to provide concise, instructive commentary explaining why a chess move was played, whether it is strong or flawed, and what tactical or positional factors drive that evaluation.

CRITICAL RULES:
1. STRICT GROUNDING: Base your explanation entirely on the provided evaluation, centipawn loss, best alternative move, and tactical theme.
2. EVAL-SIGN CONSISTENCY:
   - If the move is a blunder or mistake (positive centipawn loss), explain what was overlooked, the tactical refutation, or what the opponent will exploit. Never praise a blunder.
   - If the move is good or the best move, explain why it is energetic, principled, or defensively sound. Never describe a good move as a blunder.
3. CONCISENESS: Output exactly 2 to 4 sentences. Punchy and educational.
4. ZERO ROBOTIC FILLER: Do NOT start with phrases like "In this position", "As an AI commentator", "According to the engine", or "Here is my commentary". Jump directly into the move analysis.
5. NO HALLUCINATION: Only mention squares and pieces that actually exist in the position and are directly relevant to the tactical or strategic idea.
6. NO META-PROMPT OR INSTRUCTION LEAKAGE: Write as an expert commentator explaining the position directly — never mention that a continuation, instruction, or engine line was 'given', 'provided', 'suggested', or 'indicated' to you. State the moves and their consequences directly, as if you calculated them yourself.
"""

COMMON_OPENING_PATTERNS: Dict[tuple, str] = {
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "c7c6"): "Caro-Kann Defense (1. e4 c6)",
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "c7c5"): "Sicilian Defense (1. e4 c5)",
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "e7e6"): "French Defense (1. e4 e6)",
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "e7e5"): "King's Pawn Game (1. e4 e5)",
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "d7d5"): "Scandinavian Defense (1. e4 d5)",
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "g8f6"): "Alekhine's Defense (1. e4 Nf6)",
    ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1", "d7d5"): "Queen's Pawn Game (1. d4 d5)",
    ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1", "g8f6"): "Indian Defense (1. d4 Nf6)",
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4"): "King's Pawn Opening (1. e4)",
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "d2d4"): "Queen's Pawn Opening (1. d4)",
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "c2c4"): "English Opening (1. c4)",
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "g1f3"): "Réti Opening (1. Nf3)",
}


def _is_valid_num(val: Any) -> bool:
    if val is None:
        return False
    try:
        import math
        return not math.isnan(val)
    except (TypeError, ValueError):
        return False


def build_teacher_prompt(analysis: PositionAnalysis) -> str:
    """Format a position analysis into a structured user prompt for the Teacher LLM."""
    loss_desc = f"{int(analysis.cp_loss)} centipawns" if _is_valid_num(analysis.cp_loss) else "unknown"
    eval_before_desc = (
        f"{int(analysis.eval_before):+d} cp" if _is_valid_num(analysis.eval_before) else "even"
    )
    eval_after_desc = (
        f"{int(analysis.eval_after):+d} cp" if _is_valid_num(analysis.eval_after) else "even"
    )

    tactic = analysis.taxonomy.tactic_type
    category = analysis.taxonomy.mistake_category
    piece_lost = analysis.taxonomy.piece_lost

    best_alternative = (
        f"Recommended engine move was {analysis.best_move}."
        if not analysis.played_best and analysis.best_move
        else "The played move is the engine's top choice."
    )

    prompt = f"""Position FEN: {analysis.fen}
Move Played: {analysis.move_san} (UCI: {analysis.move_uci}) by {analysis.player_color.title()}
Game Phase: {analysis.features.phase} (Move {analysis.move_number})
Stockfish Evaluation:
- Eval Before: {eval_before_desc}
- Eval After: {eval_after_desc}
- Centipawn Loss: {loss_desc}
- Classification: {analysis.mistake_type.upper()}
- Best Move: {best_alternative}

Tactical Theme:
- Category: {category}
- Specific Motif: {tactic}
"""
    if tactic == "checkmate" or analysis.move_san.endswith("#"):
        prompt += "- OUTCOME: CHECKMATE! The move delivers checkmate and immediately ends the game.\n"

    if piece_lost != "none":
        prompt += f"- Piece Involved/Lost: {piece_lost}\n"

    # Scoped injection for high-hallucination categories
    if tactic in CONTINUATION_TACTIC_TYPES and (analysis.best_move or analysis.move_uci):
        b_cur = chess.Board(analysis.fen)
        bm_uci = analysis.best_move or analysis.move_uci
        bm_obj = chess.Move.from_uci(bm_uci) if bm_uci else None
        bm_san = b_cur.san(bm_obj) if bm_obj and bm_obj in b_cur.legal_moves else (bm_uci or "None")
        continuation_str = compute_tactical_continuation(analysis.fen, bm_uci, tactic)
        label = "Forced Mating Line (ply+1 to ply+2)" if tactic in ("missed_mate", "checkmate") else "Legal Opponent Response (ply+1)"

        prompt += f"""
Tactical Continuation:
- Engine Best Move: {bm_san}
- {label}: {continuation_str or 'None'}
Base your explanation only on the position, the move played, and the line shown above — don't introduce additional moves, checks, or captures beyond what's given here or already visible on the board.
"""
    elif tactic in VERIFICATION_TACTIC_TYPES:
        verification_block = compute_tactical_verification(
            fen=analysis.fen,
            move_uci=analysis.move_uci,
            tactic=tactic,
            player_color=analysis.player_color,
            best_move_uci=analysis.best_move,
            mistake_category=analysis.taxonomy.mistake_category,
            tactical_elements=analysis.taxonomy.tactical_elements,
        )
        if verification_block:
            prompt += f"\n{verification_block}\n"

    # Extract move nature and board dynamics using chess
    try:
        board = chess.Board(analysis.fen)

        move_obj = chess.Move.from_uci(analysis.move_uci)
        is_capture = board.is_capture(move_obj)
        is_check = board.gives_check(move_obj)
        is_castling = board.is_castling(move_obj)
        moving_piece = board.piece_at(move_obj.from_square)
        piece_name = chess.piece_name(moving_piece.piece_type).title() if moving_piece else "Piece"

        dynamics = [f"{piece_name} move"]
        if is_castling:
            dynamics.append("Castling")
        if is_capture:
            dynamics.append("Capture / Recapture")
        if is_check:
            dynamics.append("Delivers check")
        move_dynamics_desc = ", ".join(dynamics)
    except Exception:
        move_dynamics_desc = "Standard move"

    # Positional Context
    player_title = analysis.player_color.title()
    f = analysis.features
    prompt += f"""
Positional Context:
- Move Dynamics: {move_dynamics_desc}
- Mobility: {f.mobility} legal moves available
- King Safety: Castled={f.castled}, Open files near king={f.open_files_near_king}
- Pawn Structure: Doubled={f.doubled_pawns}, Isolated={f.isolated_pawns}, Passed={f.passed_pawns}
- Material on Board: Total remaining piece points={f.material_balance} ({player_title}: {f.pawns}P, {f.knights}N, {f.bishops}B, {f.rooks}R, {f.queens}Q | Opponent: {f.opp_pawns}P, {f.opp_knights}N, {f.opp_bishops}B, {f.opp_rooks}R, {f.opp_queens}Q)
- Time Pressure: {'Yes (< 60s)' if f.time_pressure else 'No'}
"""
    if f.phase == "opening":
        opening_book = COMMON_OPENING_PATTERNS.get((analysis.fen, analysis.move_uci))
        if opening_book:
            prompt += f"- Opening Theory: Known book system — {opening_book}. Explain the strategic ideas (pawn structure, central challenge, piece development) characteristic of this system.\n"
        else:
            prompt += "- Opening Context: Focus on center control, piece development, pawn structure harmony, and opening goals.\n"
    elif f.phase == "endgame":
        prompt += "- Endgame Context: Focus on king activation, passed pawn creation/advancement, and piece activity.\n"

    prompt += "\nExplain the move in 2-4 sentences with grandmaster clarity:\n"
    return prompt
