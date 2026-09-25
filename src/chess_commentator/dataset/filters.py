"""Quality filtering for synthetic commentary pairs."""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Final, Set, List, Tuple, Dict, Any
import pandas as pd


@dataclass(frozen=True)
class FilterResult:
    """Result of running a quality filter."""
    passed: bool
    reason: str = "ok"


# Banned filler or meta-prompts
BANNED_FILLER_PHRASES: Final[Set[str]] = {
    "as an ai",
    "as an artificial intelligence",
    "here is my commentary",
    "here is the commentary",
    "according to stockfish",
    "according to the engine",
    "in this chess position",
    "in this position,",
    "this is a chess move",
    "the engine suggests",
    "as a chess commentator",
    "hello chess players",
}

# Banned scaffolding or self-referential prompt leakage patterns
BANNED_SCAFFOLDING_PATTERNS: Final[List[re.Pattern]] = [
    re.compile(r"\bcritical\s+instruction\b", re.IGNORECASE),
    re.compile(r"\binstruction\s+(?:indicates|points|states|notes|requires|suggests)\b", re.IGNORECASE),
    re.compile(r"\bas\s+(?:instructed|provided\s+above|given\s+above|shown\s+above)\b", re.IGNORECASE),
    re.compile(r"\btactical\s+continuation\s+(?:shows|indicates|points|suggests)\b", re.IGNORECASE),
    re.compile(r"\b(?:the\s+)?verified\s+(?:tactical\s+)?continuation\b", re.IGNORECASE),
    re.compile(r"\bcontinuation\s+(?:provided|given|listed|above)\b", re.IGNORECASE),
    re.compile(r"\b(?:scaffolding|system\s+prompt)\b", re.IGNORECASE),
]


def validate_no_scaffolding_leakage(commentary: str) -> FilterResult:
    """Ensure commentary does not reference internal prompt scaffolding or instructions."""
    for pat in BANNED_SCAFFOLDING_PATTERNS:
        match = pat.search(commentary)
        if match:
            return FilterResult(
                passed=False,
                reason=f"Scaffolding leakage detected: '{match.group(0)}'",
            )
    return FilterResult(passed=True)


# Positive praise words that must NEVER describe a major blunder
PRAISE_WORDS: Final[Set[str]] = {
    "brilliant",
    "masterpiece",
    "excellent",
    "fantastic",
    "superb",
    "flawless",
    "great move",
    "very strong move",
    "masterful",
    "wonderful move",
    "solid improvement",
}

# Blunder words that must NEVER describe a good / best move
BLUNDER_WORDS: Final[Set[str]] = {
    "blunder",
    "terrible",
    "disastrous",
    "horrible",
    "severe error",
    "throws away",
    "catastrophic",
    "blunders away",
}


def validate_eval_sign_consistency(
    commentary: str,
    cp_loss: Optional[int],
    played_best: bool,
    mistake_type: str,
) -> FilterResult:
    """Ensure generated commentary tone does not contradict engine evaluation sign."""
    comm_lower = commentary.lower()

    # Case 1: Blunder (cp_loss >= 300 or mistake_type == 'blunder')
    is_blunder = (cp_loss is not None and cp_loss >= 300) or (mistake_type == "blunder")
    if is_blunder:
        for word in PRAISE_WORDS:
            if re.search(r'\b' + re.escape(word) + r'\b', comm_lower):
                return FilterResult(
                    passed=False,
                    reason=f"Eval-sign contradiction: Blunder described with praise ('{word}')",
                )

    # Case 2: Best move or completely good move (cp_loss <= 0 or played_best)
    is_good = played_best or (cp_loss is not None and cp_loss <= 0) or (mistake_type == "good")
    if is_good:
        for word in BLUNDER_WORDS:
            if re.search(r'\b' + re.escape(word) + r'\b', comm_lower):
                return FilterResult(
                    passed=False,
                    reason=f"Eval-sign contradiction: Good/best move described with blunder word ('{word}')",
                )

    return FilterResult(passed=True)


def validate_filler_and_length(
    commentary: str,
    min_words: int = 15,
    max_words: int = 120,
) -> FilterResult:
    """Reject robotic AI boilerplate, truncated text, or overly verbose outputs."""
    comm_lower = commentary.lower().strip()
    words = commentary.split()
    n_words = len(words)

    if n_words < min_words:
        return FilterResult(passed=False, reason=f"Too short ({n_words} words < {min_words})")

    if n_words > max_words:
        return FilterResult(passed=False, reason=f"Too long ({n_words} words > {max_words})")

    for phrase in BANNED_FILLER_PHRASES:
        if phrase in comm_lower:
            return FilterResult(
                passed=False,
                reason=f"Contains banned robotic filler: '{phrase}'",
            )

    scaffolding_check = validate_no_scaffolding_leakage(commentary)
    if not scaffolding_check.passed:
        return scaffolding_check

    return FilterResult(passed=True)


@dataclass(frozen=True)
class PhraseDiversityResult:
    """Result of phrase diversity check against bucket history."""
    is_flagged: bool
    max_overlap: float
    matched_reference: Optional[str] = None
    matched_bucket: str = "none"


class PhraseDiversityChecker:
    """Monitors n-gram overlap within taxonomy buckets to flag repetitive commentary phrasing."""

    def __init__(self, n: int = 3, threshold: float = 0.40) -> None:
        self.n = n
        self.threshold = threshold
        # bucket -> list of (commentary, set of ngrams)
        self._history: Dict[str, List[Tuple[str, Set[Tuple[str, ...]]]]] = defaultdict(list)

    @staticmethod
    def extract_ngrams(text: str, n: int = 3) -> Set[Tuple[str, ...]]:
        """Extract word n-grams from cleaned lowercase text."""
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        words = clean.split()
        if len(words) < n:
            return set()
        return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}

    def check_and_add(
        self,
        commentary: str,
        bucket: str = "none",
        add_if_flagged: bool = True,
    ) -> PhraseDiversityResult:
        """Check a commentary against prior commentaries in the same taxonomy bucket."""
        current_ngrams = self.extract_ngrams(commentary, self.n)
        if not current_ngrams:
            return PhraseDiversityResult(is_flagged=False, max_overlap=0.0, matched_bucket=bucket)

        max_overlap = 0.0
        best_match: Optional[str] = None

        for prev_comm, prev_ngrams in self._history[bucket]:
            if not prev_ngrams:
                continue
            # Containment overlap: intersection / min(len(A), len(B))
            overlap = len(current_ngrams & prev_ngrams) / min(len(current_ngrams), len(prev_ngrams))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = prev_comm

        is_flagged = max_overlap >= self.threshold

        if add_if_flagged or not is_flagged:
            self._history[bucket].append((commentary, current_ngrams))

        return PhraseDiversityResult(
            is_flagged=is_flagged,
            max_overlap=max_overlap,
            matched_reference=best_match,
            matched_bucket=bucket,
        )

    def audit_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "teacher_commentary",
        bucket_column: str = "tactic_type",
    ) -> Dict[str, Any]:
        """Audit an entire dataframe of commentary for phrase diversity within taxonomy buckets."""
        self._history.clear()
        flagged_records = []
        bucket_counts: Dict[str, int] = defaultdict(int)
        bucket_flagged: Dict[str, int] = defaultdict(int)

        for idx, row in df.iterrows():
            comm = str(row.get(text_column, ""))
            bucket = str(row.get(bucket_column, "none"))
            bucket_counts[bucket] += 1
            res = self.check_and_add(comm, bucket=bucket)
            if res.is_flagged:
                bucket_flagged[bucket] += 1
                flagged_records.append({
                    "index": idx,
                    "bucket": bucket,
                    "move": row.get("move_san", row.get("move_uci", "")),
                    "max_overlap": res.max_overlap,
                    "commentary": comm,
                    "matched_reference": res.matched_reference,
                })

        total = len(df)
        flagged_count = len(flagged_records)
        return {
            "total_checked": total,
            "flagged_count": flagged_count,
            "flag_rate": flagged_count / total if total > 0 else 0.0,
            "bucket_counts": dict(bucket_counts),
            "bucket_flagged": dict(bucket_flagged),
            "flagged_records": flagged_records,
            "threshold": self.threshold,
            "n": self.n,
        }

