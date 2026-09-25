"""Live commentary service exposing generate_commentary(fen, move)."""

import logging
from typing import Optional, Union, Dict, Any, Tuple
import chess
from chess_commentator.analysis.engine import StockfishEngine
from chess_commentator.analysis.models import PositionAnalysis, CommentaryResult
from chess_commentator.analysis.pipeline import analyze_position
from chess_commentator.dataset.formatter import SYSTEM_INSTRUCTION, construct_user_prompt
from chess_commentator.dataset.checker import validate_chess_grounding
from chess_commentator.dataset.filters import validate_eval_sign_consistency
from chess_commentator.serving.engine import CommentaryInferenceEngine

logger = logging.getLogger(__name__)


def construct_serving_prompt(
    analysis: PositionAnalysis,
    stockfish_engine: Optional[StockfishEngine] = None,
) -> str:
    """Format an analysis record into the prompt expected by the fine-tuned LLM,
    reconstructing the identical tactical continuation injection used during grounding.
    """
    user_text = construct_user_prompt(
        fen=analysis.fen,
        move_san=analysis.move_san,
        move_uci=analysis.move_uci,
        player_color=analysis.player_color,
        mistake_type=analysis.mistake_type,
        cp_loss=analysis.cp_loss,
        best_move=analysis.best_move,
        played_best=analysis.played_best,
        tactic_type=analysis.taxonomy.tactic_type,
        mistake_category=analysis.taxonomy.mistake_category,
        tactical_elements=analysis.taxonomy.tactical_elements,
        stockfish_engine=stockfish_engine,
    )
    return (
        f"<|system|>\n{SYSTEM_INSTRUCTION}<|end|>\n"
        f"<|user|>\n{user_text}<|end|>\n"
        f"<|assistant|>\n"
    )


class CommentaryService:
    """Persistent service maintaining warm Stockfish and LLM inference instances."""

    def __init__(
        self,
        stockfish_engine: Optional[StockfishEngine] = None,
        inference_engine: Optional[CommentaryInferenceEngine] = None,
        default_depth: int = 18,
        default_mode: str = "deep",
    ) -> None:
        self.stockfish = stockfish_engine or StockfishEngine(default_depth=default_depth)
        self.inference = inference_engine or CommentaryInferenceEngine()
        self.default_depth = default_depth
        self.default_mode = default_mode
        self.first_pass_rejection_count: int = 0
        self.total_generations_count: int = 0

    def _validate_completion(
        self,
        commentary: str,
        analysis: PositionAnalysis,
    ) -> Tuple[bool, bool, str]:
        """Validate completion against chess board rules and eval-sign consistency."""
        ground_res = validate_chess_grounding(
            fen=analysis.fen,
            move_uci=analysis.move_uci,
            commentary=commentary,
            best_move_uci=analysis.best_move,
        )
        eval_res = validate_eval_sign_consistency(
            commentary=commentary,
            cp_loss=analysis.cp_loss,
            played_best=analysis.played_best,
            mistake_type=analysis.mistake_type,
        )
        reasons = []
        if not ground_res.passed:
            reasons.append(f"Grounding failed: {ground_res.reason}")
        if not eval_res.passed:
            reasons.append(f"Eval consistency failed: {eval_res.reason}")
        return ground_res.passed, eval_res.passed, "; ".join(reasons)

    def _fallback_template(self, prompt: str) -> str:
        """Return deterministic fallback template commentary."""
        if hasattr(self.inference, "_heuristic_fallback"):
            return self.inference._heuristic_fallback(prompt)
        return CommentaryInferenceEngine()._heuristic_fallback(prompt)

    def comment(
        self,
        fen: str,
        move: Union[str, chess.Move],
        depth: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> CommentaryResult:
        """Analyze a move and generate grounded commentary."""
        analysis = analyze_position(
            fen=fen,
            move=move,
            engine=self.stockfish,
            depth=depth or self.default_depth,
            mode=mode or self.default_mode,
        )

        prompt = construct_serving_prompt(analysis, stockfish_engine=self.stockfish)
        self.total_generations_count += 1
        commentary_text = self.inference.generate(prompt)

        # Stage 6 Guardrail Validation: Check grounding and eval-sign consistency
        is_grounded, is_eval_consistent, reason = self._validate_completion(commentary_text, analysis)
        if not (is_grounded and is_eval_consistent):
            self.first_pass_rejection_count += 1
            rejection_pct = (self.first_pass_rejection_count / self.total_generations_count) * 100.0
            logger.warning(
                "First-pass commentary rejected by Stage 6 guardrail: %s. "
                "Total first-pass rejections: %d/%d (%.1f%%). Attempting regeneration...",
                reason,
                self.first_pass_rejection_count,
                self.total_generations_count,
                rejection_pct,
            )
            # 1. Regeneration attempt: re-sample with small temperature bump
            regenerated_text = self.inference.generate(prompt, temperature=0.3)
            is_regen_grounded, is_regen_eval, regen_reason = self._validate_completion(regenerated_text, analysis)

            if is_regen_grounded and is_regen_eval:
                logger.info("Regenerated commentary passed Stage 6 guardrail validation.")
                commentary_text = regenerated_text
            else:
                # 2. If regenerated completion also fails, return fallback/template response
                logger.warning(
                    "Regenerated commentary also failed guardrail validation: %s. "
                    "Returning grounded heuristic fallback template.",
                    regen_reason,
                )
                commentary_text = self._fallback_template(prompt)

        return CommentaryResult(
            fen=analysis.fen,
            move=analysis.move_san,
            commentary=commentary_text,
            taxonomy=analysis.taxonomy.tactic_type,
            cp_loss=analysis.cp_loss,
            eval_before=analysis.eval_before,
            eval_after=analysis.eval_after,
            best_move=analysis.best_move,
            is_blunder=(analysis.mistake_type == "blunder"),
            phase=analysis.features.phase,
            grounded_features=analysis.features.to_dict(),
        )

    def close(self) -> None:
        """Release underlying resources."""
        self.stockfish.close()


def generate_commentary(
    fen: str,
    move: Union[str, chess.Move],
    engine: Optional[StockfishEngine] = None,
    depth: int = 18,
    mode: str = "deep",
    adapter_path: Optional[str] = None,
    base_model_id: str = "microsoft/Phi-3.5-mini-instruct",
    inference_engine: Optional[CommentaryInferenceEngine] = None,
) -> CommentaryResult:
    """Serve live natural-language chess commentary explaining why a move is good or bad.

    Args:
        fen: FEN board position string before the move.
        move: Move in UCI (e.g. 'e2e4') or SAN (e.g. 'e4') format.
        engine: Optional pre-configured StockfishEngine.
        depth: Stockfish search depth (default: 18).
        mode: Analysis mode ('deep' or 'fast').
        adapter_path: Path to fine-tuned QLoRA adapter weights.
        base_model_id: Base Hugging Face model identifier.
        inference_engine: Optional pre-loaded CommentaryInferenceEngine.

    Returns:
        CommentaryResult containing commentary text, taxonomy, and engine statistics.
    """
    inf_engine = inference_engine or CommentaryInferenceEngine(
        base_model_id=base_model_id,
        adapter_path=adapter_path,
    )
    service = CommentaryService(
        stockfish_engine=engine,
        inference_engine=inf_engine,
        default_depth=depth,
        default_mode=mode,
    )
    should_close_stockfish = (engine is None)

    try:
        return service.comment(fen=fen, move=move, depth=depth, mode=mode)
    finally:
        if should_close_stockfish:
            service.close()
