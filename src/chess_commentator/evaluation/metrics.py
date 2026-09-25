"""Automated NLP and chess-specific evaluation metrics."""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import re
from chess_commentator.dataset.filters import validate_eval_sign_consistency
from chess_commentator.dataset.checker import validate_chess_grounding


@dataclass(frozen=True)
class EvaluationSummary:
    """Summary of all automated evaluation metrics across a dataset."""
    rouge_l: float
    token_f1: float
    eval_sentiment_agreement: float
    tactic_recall: float
    hallucination_rate: float
    total_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _tokenize(text: str) -> List[str]:
    """Simple lowercase word tokenization."""
    return re.findall(r'\b\w+\b', text.lower())


def _lcs(x: List[str], y: List[str]) -> int:
    """Longest common subsequence between two token lists."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[m][n]


def compute_rouge_l_single(pred: str, ref: str) -> float:
    """Compute ROUGE-L F1 score for a single prediction and reference pair."""
    p_tokens = _tokenize(pred)
    r_tokens = _tokenize(ref)
    if not p_tokens or not r_tokens:
        return 0.0
    lcs_len = _lcs(p_tokens, r_tokens)
    prec = lcs_len / len(p_tokens)
    rec = lcs_len / len(r_tokens)
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def compute_token_f1_single(pred: str, ref: str) -> float:
    """Compute unigram F1 score between prediction and reference."""
    p_set = set(_tokenize(pred))
    r_set = set(_tokenize(ref))
    if not p_set or not r_set:
        return 0.0
    common = len(p_set & r_set)
    if common == 0:
        return 0.0
    prec = common / len(p_set)
    rec = common / len(r_set)
    return (2 * prec * rec) / (prec + rec)


def compute_text_metrics(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Compute mean ROUGE-L and Token-F1 over lists of predictions and references."""
    if not predictions or not references:
        return {"rouge_l": 0.0, "token_f1": 0.0}

    rouge_scores = [
        compute_rouge_l_single(p, r) for p, r in zip(predictions, references)
    ]
    f1_scores = [
        compute_token_f1_single(p, r) for p, r in zip(predictions, references)
    ]

    return {
        "rouge_l": round(sum(rouge_scores) / len(rouge_scores), 4),
        "token_f1": round(sum(f1_scores) / len(f1_scores), 4),
    }


def compute_eval_sentiment_agreement(
    predictions: List[str],
    mistake_types: List[str],
    cp_losses: Optional[List[Optional[int]]] = None,
) -> float:
    """Calculate percentage of predictions whose tone matches the engine evaluation."""
    if not predictions:
        return 0.0

    agreed = 0
    total = len(predictions)

    for i in range(total):
        pred = predictions[i]
        m_type = mistake_types[i]
        loss = cp_losses[i] if cp_losses else (300 if m_type == "blunder" else 0)
        played_best = (m_type == "good")

        check = validate_eval_sign_consistency(
            commentary=pred,
            cp_loss=loss,
            played_best=played_best,
            mistake_type=m_type,
        )
        if check.passed:
            agreed += 1

    return round(agreed / total, 4)


def compute_tactic_recall(
    predictions: List[str], tactic_types: List[str]
) -> float:
    """Calculate percentage of tactical positions where the theme was recalled."""
    tactical_indices = [
        i for i, t in enumerate(tactic_types)
        if t not in ("none", "unknown", "opening_principle", "endgame_technique")
    ]
    if not tactical_indices:
        return 1.0

    recalled = 0
    for idx in tactical_indices:
        tactic = tactic_types[idx].lower().replace("_", " ")
        pred = predictions[idx].lower()

        # Check direct mention of tactic or base keyword
        base_words = tactic.split()
        if any(w in pred for w in base_words if len(w) > 3):
            recalled += 1

    return round(recalled / len(tactical_indices), 4)


def compute_hallucination_rate(
    predictions: List[str],
    fens: List[str],
    moves: List[str],
) -> float:
    """Percentage of predictions that contain hallucinated piece/square combinations."""
    if not predictions:
        return 0.0

    hallucinated = 0
    total = len(predictions)

    for pred, fen, move in zip(predictions, fens, moves):
        check = validate_chess_grounding(fen=fen, move_uci=move, commentary=pred)
        if not check.passed:
            hallucinated += 1

    return round(hallucinated / total, 4)
