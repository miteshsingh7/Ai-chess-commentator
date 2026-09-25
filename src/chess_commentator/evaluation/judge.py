"""LLM-as-a-Judge evaluation framework for chess commentary quality."""

import json
import re
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from chess_commentator.teacher.client import ClaudeTeacherClient

JUDGE_RUBRIC_PROMPT = """You are an International Arbiter and Chess Grandmaster evaluating the quality of generated chess commentary.
Grade the candidate commentary against the ground-truth chess position, move, Stockfish evaluation, and tactical theme.

Score each dimension from 1 to 5 (integer):
1. ACCURACY (1-5): Is the commentary factually accurate? Does it match the evaluation sign (praising good moves, identifying blunders)? Are piece and square references correct?
2. EDUCATIONAL_VALUE (1-5): Does the commentary explain WHY the move succeeds or fails? Does it highlight the tactical theme or strategic idea?
3. NATURALNESS (1-5): Does it sound like an authentic grandmaster commentary? Is the phrasing natural and engaging?
4. CONCISENESS (1-5): Is it punchy (2-4 sentences) with zero robotic filler or repetitive phrases?

Output ONLY a valid JSON object formatted as follows:
{
  "accuracy": <1-5>,
  "educational_value": <1-5>,
  "naturalness": <1-5>,
  "conciseness": <1-5>,
  "feedback": "<1-2 sentence constructive critique>"
}
"""


@dataclass(frozen=True)
class JudgeScore:
    """Evaluation score awarded by LLM Judge."""
    accuracy: int
    educational_value: int
    naturalness: int
    conciseness: int
    overall: float
    feedback: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMJudge:
    """Evaluates candidate commentary using Claude or mock judge."""

    def __init__(
        self,
        client: Optional[ClaudeTeacherClient] = None,
        mock_mode: bool = False,
    ) -> None:
        self.client = client or ClaudeTeacherClient(mock_mode=mock_mode)
        self.mock_mode = mock_mode

    def evaluate(
        self,
        fen: str,
        move: str,
        eval_summary: str,
        tactical_motif: str,
        candidate_commentary: str,
        reference_commentary: Optional[str] = None,
    ) -> JudgeScore:
        """Score candidate commentary on the 1-5 rubric."""
        if self.mock_mode or self.client.mock_mode:
            # Deterministic heuristic scoring for mock/offline runs
            return self._mock_score(candidate_commentary, eval_summary)

        user_content = f"""Position FEN: {fen}
Move: {move}
Evaluation Summary: {eval_summary}
Tactical Motif: {tactical_motif}
Reference Teacher Commentary: {reference_commentary or 'N/A'}

Candidate Commentary to Evaluate:
\"\"\"{candidate_commentary}\"\"\"
"""
        response_text = self.client.generate(f"{JUDGE_RUBRIC_PROMPT}\n\n{user_content}")
        return self._parse_judge_json(response_text)

    def _mock_score(self, commentary: str, eval_summary: str) -> JudgeScore:
        """Heuristic mock score generator."""
        words = len(commentary.split())
        acc = 5
        edu = 4
        nat = 5
        conc = 5 if (15 <= words <= 80) else 3

        if "blunder" in eval_summary.lower() and "brilliant" in commentary.lower():
            acc = 1
            edu = 1

        overall = round((acc + edu + nat + conc) / 4.0, 2)
        return JudgeScore(
            accuracy=acc,
            educational_value=edu,
            naturalness=nat,
            conciseness=conc,
            overall=overall,
            feedback="Clear explanation with accurate tactical grounding.",
        )

    @staticmethod
    def _parse_judge_json(text: str) -> JudgeScore:
        """Parse JSON score object from LLM response."""
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                acc = int(data.get("accuracy", 3))
                edu = int(data.get("educational_value", 3))
                nat = int(data.get("naturalness", 3))
                conc = int(data.get("conciseness", 3))
                feedback = str(data.get("feedback", ""))
                overall = round((acc + edu + nat + conc) / 4.0, 2)
                return JudgeScore(
                    accuracy=acc,
                    educational_value=edu,
                    naturalness=nat,
                    conciseness=conc,
                    overall=overall,
                    feedback=feedback,
                )
        except Exception:
            pass

        return JudgeScore(
            accuracy=3,
            educational_value=3,
            naturalness=3,
            conciseness=3,
            overall=3.0,
            feedback="Parsing failed, defaulted to neutral score.",
        )
