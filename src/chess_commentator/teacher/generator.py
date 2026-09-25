"""Batch generator orchestrating Teacher LLM commentary creation with routing and caching."""

import os
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
from tqdm import tqdm

from chess_commentator.analysis.models import PositionAnalysis
from chess_commentator.teacher.cache import TeacherCache
from chess_commentator.teacher.client import ClaudeTeacherClient
from chess_commentator.teacher.prompt_builder import build_teacher_prompt

GROQ_HEAVY_MODEL = "openai/gpt-oss-120b"
GROQ_FAST_MODEL = "qwen/qwen3.8-27b"

# Backward compatibility aliases
SONNET_MODEL = GROQ_HEAVY_MODEL
HAIKU_MODEL = GROQ_FAST_MODEL

CRITICAL_TACTIC_TYPES = {
    "missed_forced_mate",
    "missed_mate",
    "missed_sacrifice",
    "checkmate",
}


def route_teacher_model(analysis: PositionAnalysis) -> Tuple[str, str]:
    """Determine whether to use Groq Heavy (120B reasoning) or Groq Fast (27B).
    
    Routes to Groq Heavy if:
      - cp_loss >= 300 (major blunder)
      - tactic_type in {'missed_forced_mate', 'missed_mate', 'missed_sacrifice', 'checkmate'}
    Otherwise routes to Groq Fast for cost efficiency and low latency.
    """
    raw_loss = analysis.cp_loss
    loss = 0
    if raw_loss is not None:
        try:
            import math
            if not math.isnan(raw_loss):
                loss = int(raw_loss)
        except (TypeError, ValueError):
            pass

    tactic = analysis.taxonomy.tactic_type.lower()

    if loss >= 300:
        return GROQ_HEAVY_MODEL, f"High severity blunder (cp_loss={loss} >= 300)"
    if tactic in CRITICAL_TACTIC_TYPES:
        return GROQ_HEAVY_MODEL, f"Critical tactical motif ('{tactic}')"
    return GROQ_FAST_MODEL, "Standard positional / tactical evaluation"


class TeacherCommentaryGenerator:
    """Orchestrates generation of teacher commentary across positions with model routing."""

    def __init__(
        self,
        client: Optional[ClaudeTeacherClient] = None,
        cache: Optional[TeacherCache] = None,
        cache_dir: str = ".cache/teacher_commentary",
        enable_routing: bool = True,
    ) -> None:
        self.client = client or ClaudeTeacherClient()
        self.cache = cache or TeacherCache(cache_dir=cache_dir)
        self.enable_routing = enable_routing

    def generate_for_position(
        self,
        analysis: PositionAnalysis,
        model_override: Optional[str] = None,
        temperature: Optional[float] = None,
        nonce: Optional[Any] = None,
    ) -> Tuple[str, str, str]:
        """Generate or retrieve cached commentary for a single PositionAnalysis.
        
        Returns:
            (commentary, model_used, routing_reason)
        """
        if model_override:
            target_model = model_override
            reason = "Manual override"
        elif self.enable_routing:
            target_model, reason = route_teacher_model(analysis)
        else:
            target_model = self.client.model
            reason = "Default model"

        content_hash = self.cache.compute_hash(
            fen=analysis.fen,
            move=analysis.move_uci,
            cp_loss=analysis.cp_loss,
            taxonomy=analysis.taxonomy.tactic_type,
            model=target_model,
            nonce=nonce,
        )

        cached_text = self.cache.get(content_hash)
        if cached_text:
            return cached_text, target_model, f"{reason} (cached)"

        prompt = build_teacher_prompt(analysis)
        commentary = self.client.generate(prompt, model=target_model, temperature=temperature)

        self.cache.set(
            content_hash=content_hash,
            commentary=commentary,
            metadata={
                "fen": analysis.fen,
                "move_san": analysis.move_san,
                "taxonomy": analysis.taxonomy.tactic_type,
                "mistake_type": analysis.mistake_type,
                "model": target_model,
                "reason": reason,
                "nonce": nonce,
            },
        )
        return commentary, target_model, reason

    def generate_batch(
        self,
        analyses: List[PositionAnalysis],
        output_parquet: Optional[str] = None,
        model_override: Optional[str] = None,
        temperature: Optional[float] = None,
        nonces: Optional[List[Any]] = None,
    ) -> pd.DataFrame:
        """Generate commentary for a list of PositionAnalysis objects."""
        rows: List[Dict[str, Any]] = []

        for idx, item in enumerate(tqdm(analyses, desc="Generating Teacher Commentary")):
            nonce = nonces[idx] if nonces and idx < len(nonces) else None
            comm, model_used, reason = self.generate_for_position(
                item, model_override=model_override, temperature=temperature, nonce=nonce
            )
            record = item.to_dict()
            record["teacher_commentary"] = comm
            record["teacher_model"] = model_used
            record["teacher_routing_reason"] = reason
            rows.append(record)

        df = pd.DataFrame(rows)
        if output_parquet and not df.empty:
            dir_name = os.path.dirname(output_parquet)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            df.to_parquet(output_parquet, index=False)

        return df
