import json
import os
from typing import Dict, Any, List, Optional
import chess
import pandas as pd

from chess_commentator.analysis.engine import StockfishEngine
from chess_commentator.teacher.prompt_builder import (
    CONTINUATION_TACTIC_TYPES,
    VERIFICATION_TACTIC_TYPES,
    compute_tactical_continuation,
    compute_tactical_verification,
)

SYSTEM_INSTRUCTION = """You are a grandmaster chess commentator. Provide concise, 2-4 sentence expert commentary explaining why the move was played, whether it is strong or flawed, and its tactical/strategic justification. State the moves and consequences directly as your own grandmaster calculation; never refer to prompts, instructions, or provided lines."""


def construct_user_prompt(
    fen: str,
    move_san: str,
    move_uci: str,
    player_color: str,
    mistake_type: str,
    cp_loss: Optional[int],
    best_move: Optional[str] = None,
    played_best: bool = False,
    tactic_type: str = "none",
    mistake_category: str = "none",
    tactical_elements: Optional[Dict[str, Any]] = None,
    stockfish_engine: Optional[StockfishEngine] = None,
) -> str:
    """Format position, move, engine evaluation, and optional tactical continuation into standard user prompt text."""
    loss_str = f"loss: {cp_loss} cp" if cp_loss is not None else "even"
    best_str = (
        f", best move was {best_move}"
        if not played_best and best_move and best_move != move_uci
        else ""
    )

    motif = "checkmate" if move_san.endswith("#") else tactic_type

    user_text = (
        f"Position: {fen}\n"
        f"Move: {move_san} ({move_uci}) by {player_color.title()}\n"
        f"Engine: {mistake_type.upper()} ({loss_str}{best_str})\n"
        f"Tactical Motif: {motif}"
    )

    active_tactic = (
        motif
        if motif in CONTINUATION_TACTIC_TYPES
        else (mistake_category if mistake_category in CONTINUATION_TACTIC_TYPES else None)
    )

    active_verification = (
        motif
        if motif in VERIFICATION_TACTIC_TYPES
        else (mistake_category if mistake_category in VERIFICATION_TACTIC_TYPES else None)
    )

    if active_tactic and (best_move or move_uci):
        b_cur = chess.Board(fen)
        bm_uci = best_move or move_uci
        bm_obj = chess.Move.from_uci(bm_uci) if bm_uci else None
        bm_san = b_cur.san(bm_obj) if bm_obj and bm_obj in b_cur.legal_moves else (bm_uci or "None")
        continuation_str = compute_tactical_continuation(
            fen, bm_uci, active_tactic, engine=stockfish_engine
        )
        label = (
            "Forced Mating Line (ply+1 to ply+2)"
            if active_tactic in ("missed_mate", "checkmate")
            else "Legal Opponent Response (ply+1)"
        )

        user_text += (
            f"\n\nTactical Continuation:\n"
            f"- Engine Best Move: {bm_san}\n"
            f"- {label}: {continuation_str or 'None'}\n"
            f"Base your explanation only on the position, the move played, and the line shown above — "
            f"don't introduce additional moves, checks, or captures beyond what's given here or already visible on the board."
        )
    elif active_verification:
        verification_block = compute_tactical_verification(
            fen=fen,
            move_uci=move_uci,
            tactic=active_verification,
            player_color=player_color,
            best_move_uci=best_move,
            mistake_category=mistake_category,
            tactical_elements=tactical_elements,
        )
        if verification_block:
            user_text += f"\n\n{verification_block}"

    return user_text


def row_to_chatml(
    row: Dict[str, Any],
    stockfish_engine: Optional[StockfishEngine] = None,
) -> Dict[str, Any]:
    """Convert a dataset row into standard ChatML conversation structure."""
    fen = row.get("fen", "")
    move_san = row.get("move_san", "")
    move_uci = row.get("move_uci", "")
    player_color = row.get("player_color", "white")
    cp_loss = row.get("cp_loss")
    mistake_type = row.get("mistake_type", "good")
    tactic = row.get("tactic_type", "none")
    mistake_category = row.get("mistake_category", "none")
    best_move = row.get("best_move")
    played_best = (
        row.get("played_best", False)
        if "played_best" in row and row["played_best"] is not None
        else (best_move == move_uci if best_move else False)
    )
    commentary = row.get("teacher_commentary", "") or row.get("commentary", "")

    user_text = construct_user_prompt(
        fen=fen,
        move_san=move_san,
        move_uci=move_uci,
        player_color=player_color,
        mistake_type=mistake_type,
        cp_loss=cp_loss,
        best_move=best_move,
        played_best=bool(played_best),
        tactic_type=tactic,
        mistake_category=mistake_category,
        tactical_elements=row.get("tactical_elements"),
        stockfish_engine=stockfish_engine,
    )

    full_text = (
        f"<|system|>\n{SYSTEM_INSTRUCTION}<|end|>\n"
        f"<|user|>\n{user_text}<|end|>\n"
        f"<|assistant|>\n{commentary}<|end|>\n"
    )

    return {
        "text": full_text,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": commentary},
        ],
    }


def format_to_chatml(
    df: pd.DataFrame,
    stockfish_engine: Optional[StockfishEngine] = None,
) -> List[Dict[str, Any]]:
    """Convert entire DataFrame to list of ChatML dicts."""
    records = df.to_dict("records")
    return [row_to_chatml(r, stockfish_engine=stockfish_engine) for r in records]


def export_sft_dataset(
    df: pd.DataFrame,
    output_jsonl: str,
    stockfish_engine: Optional[StockfishEngine] = None,
) -> None:
    """Export formatted dataset directly to JSONL file."""
    chatml_records = format_to_chatml(df, stockfish_engine=stockfish_engine)
    dir_name = os.path.dirname(output_jsonl)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in chatml_records:
            f.write(json.dumps(item) + "\n")
