"""Groq API client via OpenAI-compatible chat completions with rate limiting, spend tracking, and mock fallback."""

import collections
import logging
import os
import time
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

from chess_commentator.teacher.prompt_builder import TEACHER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class RateLimiter:
    """Sliding-window rate limiter to cap requests per minute (default: 25 RPM)."""

    def __init__(self, max_requests: int = 25, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: collections.deque = collections.deque()

    def acquire(self) -> int:
        """Wait until a request slot is available and return remaining quota."""
        now = time.time()
        while self._timestamps and self._timestamps[0] <= now - self.window_seconds:
            self._timestamps.popleft()

        if len(self._timestamps) >= self.max_requests:
            oldest = self._timestamps[0]
            sleep_needed = (oldest + self.window_seconds) - now + 0.05
            if sleep_needed > 0:
                time.sleep(sleep_needed)
                now = time.time()
                while self._timestamps and self._timestamps[0] <= now - self.window_seconds:
                    self._timestamps.popleft()

        self._timestamps.append(time.time())
        return max(0, self.max_requests - len(self._timestamps))

    @property
    def remaining(self) -> int:
        now = time.time()
        while self._timestamps and self._timestamps[0] <= now - self.window_seconds:
            self._timestamps.popleft()
        return max(0, self.max_requests - len(self._timestamps))


class GroqTeacherClient:
    """Client for generating commentary using Groq's OpenAI-compatible API with rate limiting."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen/qwen3.8-27b",
        temperature: float = 0.3,
        max_tokens: int = 1500,
        mock_mode: bool = False,
        requests_per_minute: int = 25,
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.default_model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mock_mode = mock_mode
        self._client = None
        self.rate_limiter = RateLimiter(max_requests=requests_per_minute, window_seconds=60.0)

        # Usage and cost tracking
        self.call_history: List[Dict[str, Any]] = []

        if not self.mock_mode:
            if not self.api_key:
                raise ValueError(
                    "GROQ_API_KEY is not set and mock_mode=False. "
                    "Set GROQ_API_KEY in .env or your environment, or explicitly pass mock_mode=True for offline testing."
                )
            try:
                import openai
                self._client = openai.OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=self.api_key,
                )
            except ImportError as e:
                raise ImportError(
                    "The 'openai' package is required when mock_mode=False. Run 'pip install openai'."
                ) from e

    @property
    def model(self) -> str:
        return self.default_model

    def generate(
        self,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: int = 4,
    ) -> str:
        """Call Groq to generate grounded commentary with rate limiting and exponential backoff."""
        target_model = model or self.default_model
        eff_temperature = temperature if temperature is not None else self.temperature

        if self.mock_mode:
            # Deterministic mock response for testing/offline environments
            comm = self._generate_mock_commentary(user_prompt)
            self.call_history.append({
                "model": f"{target_model} [mock (no API call)]",
                "input_tokens": len(user_prompt.split()) * 2,
                "output_tokens": len(comm.split()) * 2,
                "cost_usd": 0.0,
                "is_mock": True,
            })
            return comm

        if self._client is None:
            raise RuntimeError("Groq API client is not initialized.")

        delay = 2.0
        last_exception = None

        for attempt in range(max_retries):
            # Enforce 25 RPM sliding-window rate limit
            remaining_quota = self.rate_limiter.acquire()
            print(f"  ⚡ [Groq API] Model: {target_model} | Quota: {remaining_quota}/25 left in 60s window")
            logger.info(f"RateLimiter: {remaining_quota}/25 requests remaining in current 60s window")

            # Set reasonable token limit to avoid pre-allocation limit errors on free tiers
            token_limit = max(self.max_tokens, 1200) if "120b" in target_model else self.max_tokens

            try:
                response = self._client.chat.completions.create(
                    model=target_model,
                    max_tokens=token_limit,
                    temperature=eff_temperature,
                    messages=[
                        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                commentary = response.choices[0].message.content or ""
                commentary = commentary.strip()

                if not commentary:
                    finish_reason = response.choices[0].finish_reason
                    raise RuntimeError(
                        f"Groq model '{target_model}' returned empty content (finish_reason='{finish_reason}'). "
                        "Reasoning tokens may have exhausted the token limit. Retrying..."
                    )

                in_tok = response.usage.prompt_tokens if response.usage else len(user_prompt.split()) * 2
                out_tok = response.usage.completion_tokens if response.usage else len(commentary.split()) * 2

                self.call_history.append({
                    "model": target_model,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cost_usd": 0.0,
                    "is_mock": False,
                })

                return commentary
            except Exception as e:
                last_exception = e
                err_str = str(e).lower()
                is_429 = "429" in err_str or getattr(e, "status_code", None) == 429
                is_tpd = "tokens per day" in err_str or "tpd" in err_str

                if is_tpd:
                    if target_model != "qwen/qwen3.8-27b":
                        print(f"  ⚠️ Daily token limit reached on {target_model}. Falling back to qwen/qwen3.8-27b...")
                        self.default_model = "qwen/qwen3.8-27b"
                        target_model = "qwen/qwen3.8-27b"
                        continue
                    else:
                        print("  ⚠️ Daily token limit reached on qwen/qwen3.8-27b (TPD exhausted). Activating grounded fallback generator...")
                        self.mock_mode = True
                        comm = self._generate_mock_commentary(user_prompt)
                        self.call_history.append({
                            "model": f"{target_model} [fallback-grounded]",
                            "input_tokens": len(user_prompt.split()) * 2,
                            "output_tokens": len(comm.split()) * 2,
                            "cost_usd": 0.0,
                            "is_mock": True,
                        })
                        return comm

                if is_429:
                    logger.warning(
                        f"Groq 429 rate limit reached. Backing off for {delay:.1f}s (attempt {attempt + 1}/{max_retries})..."
                    )
                else:
                    logger.warning(f"Groq API error: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2.0

        raise RuntimeError(f"Groq API failed after {max_retries} attempts: {last_exception}")

    def get_spend_summary(self) -> Dict[str, Any]:
        """Aggregate total token usage and financial cost across calls, separating real from mock."""
        total_calls = len(self.call_history)
        real_calls = sum(1 for c in self.call_history if not c.get("is_mock"))
        mock_calls = sum(1 for c in self.call_history if c.get("is_mock"))
        real_in_tokens = sum(c.get("input_tokens", 0) for c in self.call_history if not c.get("is_mock"))
        real_out_tokens = sum(c.get("output_tokens", 0) for c in self.call_history if not c.get("is_mock"))
        real_cost = sum(c.get("cost_usd", 0.0) for c in self.call_history if not c.get("is_mock"))

        by_model_real: Dict[str, Dict[str, Any]] = {}
        by_model_mock: Dict[str, Dict[str, Any]] = {}

        for c in self.call_history:
            m = c["model"]
            if c.get("is_mock"):
                if m not in by_model_mock:
                    by_model_mock[m] = {
                        "calls": 0,
                        "simulated_in_tokens": 0,
                        "simulated_out_tokens": 0,
                        "cost_usd": 0.0,
                    }
                by_model_mock[m]["calls"] += 1
                by_model_mock[m]["simulated_in_tokens"] += c.get("input_tokens", 0)
                by_model_mock[m]["simulated_out_tokens"] += c.get("output_tokens", 0)
            else:
                if m not in by_model_real:
                    by_model_real[m] = {"calls": 0, "in_tokens": 0, "out_tokens": 0, "cost_usd": 0.0}
                by_model_real[m]["calls"] += 1
                by_model_real[m]["in_tokens"] += c.get("input_tokens", 0)
                by_model_real[m]["out_tokens"] += c.get("output_tokens", 0)
                by_model_real[m]["cost_usd"] += c.get("cost_usd", 0.0)

        return {
            "total_calls": total_calls,
            "real_api_calls": real_calls,
            "mock_calls": mock_calls,
            "total_input_tokens": real_in_tokens,
            "total_output_tokens": real_out_tokens,
            "total_cost_usd": round(real_cost, 6),
            "by_model": by_model_real,
            "mock_breakdown": by_model_mock,
        }


    def _generate_mock_commentary(self, user_prompt: str) -> str:
        """Generate a realistic, diverse, grounded commentary based on prompt features."""
        import hashlib
        h = int(hashlib.md5(user_prompt.encode("utf-8")).hexdigest(), 16)

        lines = user_prompt.splitlines()
        move_line = next((l for l in lines if l.startswith("Move Played:")), "Move Played: e4")
        move = move_line.split(":")[1].split("(")[0].strip()

        # Extract features
        phase = "middlegame"
        if "opening" in user_prompt.lower():
            phase = "opening"
        elif "endgame" in user_prompt.lower():
            phase = "endgame"

        loss = "tactical loss"
        for l in lines:
            if "Centipawn Loss:" in l:
                loss = l.split(":")[1].strip()
                break

        best_move = ""
        for l in lines:
            if "Engine Best Move:" in l:
                best_move = l.split(":")[1].strip()
                break
            elif "Best Move:" in l and "was" in l:
                best_move = l.split("was")[-1].strip().rstrip(".")

        tactic = "tactical motif"
        for l in lines:
            if "Specific Motif:" in l:
                tactic = l.split(":")[1].strip()
                break

        # Checkmate
        if "CHECKMATE" in user_prompt or "#" in move:
            openings = [
                f"Checkmate! {move} delivers an immediate and game-ending blow against the helpless king.",
                f"A lethal finish. {move} seals the game instantly by forcing checkmate on the board.",
                f"The final strike. {move} concludes the struggle by trapping the enemy monarch with no escape.",
                f"Decisive execution! {move} lands the mating blow and brings the contest to a swift close.",
                f"Game over. {move} demonstrates flawless endgame technique, finishing with an inescapable checkmate.",
            ]
            closings = [
                "With all flight squares covered, the opponent has no legal recourse to avert defeat.",
                "The coordinated geometry leaves no squares for the king and ends the battle on the spot.",
                "Every defensive interposition is eliminated, securing an immediate full point.",
                "A clean tactical conversion that rewards the preceding positional buildup.",
            ]
            return f"{openings[h % len(openings)]} {closings[(h >> 4) % len(closings)]}"

        # Blunder / Mistake
        if "BLUNDER" in user_prompt or "MISTAKE" in user_prompt:
            alt_clause = (
                f" The engine identifies {best_move} as the crucial resource to preserve the position."
                if best_move and best_move != "None"
                else " A more resilient defensive setup was required."
            )

            if "hanging" in tactic or "hanging_piece" in user_prompt:
                openings = [
                    f"A critical oversight in the {phase}. {move} carelessly leaves an essential piece en prise without sufficient protection.",
                    f"{move} represents a severe blunder, abandoning key piece coordination and allowing a free capture.",
                    f"A painful tactical slip. By playing {move}, a vital defender is compromised and left completely hanging.",
                    f"{move} drops material directly, overlooking the immediate vulnerability created on the board.",
                ]
            elif "fork" in tactic:
                openings = [
                    f"A sharp miscalculation. {move} walks right into a tactical fork that forces decisive material concessions.",
                    f"{move} fails to anticipate the opponent's dual attack, permitting a tactical strike that forks two key targets.",
                    f"A costly tactical blunder. {move} allows a devastating fork that shatters the defensive structure.",
                    f"By playing {move}, the position is exposed to an immediate double attack that wins valuable material.",
                ]
            elif "overloaded" in tactic:
                openings = [
                    f"{move} neglects the heavy defensive responsibilities of the position, succumbing to an overload tactic.",
                    f"A tactical mistake. {move} asks a single piece to guard too many duties, enabling a decisive breakthrough.",
                    f"By committing {move}, the defensive burden collapses as the opponent exploits the overloaded defenders.",
                    f"{move} fails to manage piece pressure, allowing the opponent to strike at an overloaded sector of the board.",
                ]
            elif "missed_mate" in tactic:
                openings = [
                    f"{move} squanders a forced mating sequence, allowing the enemy monarch to slip away from danger.",
                    f"A catastrophic tactical miss. {move} forfeits an immediate mating net that was ready to close.",
                    f"Instead of finishing the contest with a forced mate, {move} lets the king escape to safety.",
                    f"{move} overlooks the decisive mating combination, turning a crushing victory into a complicated battle.",
                ]
            elif "sacrifice" in tactic:
                openings = [
                    f"{move} fails to find the decisive sacrifice, missing an opportunity to break through the defenses.",
                    f"A major tactical omission. {move} shies away from a dynamic piece sacrifice that would have sealed the win.",
                    f"{move} plays too passively, overlooking a crushing tactical sacrifice that shatters the opponent's guard.",
                ]
            elif "zwischenzug" in tactic:
                openings = [
                    f"{move} overlooks a critical in-between move, surrendering the initiative at a crucial moment.",
                    f"A tactical blunder. {move} fails to anticipate an intermediate thrust that alters the calculation.",
                    f"By missing the possibility of a zwischenzug, {move} hands the opponent a decisive tactical resource.",
                ]
            else:
                openings = [
                    f"{move} misjudges the concrete demands of the {phase}, conceding a significant advantage of {loss}.",
                    f"A costly inaccuracy. {move} compromises piece harmony and invites an immediate tactical counter.",
                    f"{move} relinquishes control of the position, allowing the opponent to exploit concrete weaknesses.",
                    f"An unfortunate misstep. {move} hands the initiative to the opponent and worsens the evaluation considerably.",
                ]

            middles = [
                f"The resulting sequence costs roughly {loss} and gives the opponent a clear path forward.",
                f"This mistake significantly diminishes the evaluation and surrenders vital central control.",
                f"The tactical refutation is swift, severely compromising the structural integrity of the position.",
                f"This oversight drastically swings the momentum and leaves the king or pieces dangerously exposed.",
            ]
            return f"{openings[h % len(openings)]} {middles[(h >> 3) % len(middles)]}{alt_clause}"

        # Good / Best moves
        openings = [
            f"A sharp and principled choice. {move} seizes immediate central space and restricts the opponent's counterplay.",
            f"{move} is an energetic, high-level move that strengthens piece coordination and maintains a firm positional grip.",
            f"An exemplary move in the {phase}. {move} proactively addresses tactical threats while improving active piece mobility.",
            f"{move} demonstrates precise calculation, consolidating the position and pressing the advantage smoothly.",
            f"A calm, master-level decision. {move} reinforces key central squares and prepares harmonious long-term development.",
            f"{move} is top-tier play, combining defensive solidity with purposeful piece activation across the board.",
        ]
        closings = [
            "It maintains harmonious board control and keeps the opponent under continuous pressure.",
            "This principled advance cements a durable advantage heading into the next phase of the game.",
            "By coordinating the pieces efficiently, this maneuver neutralizes potential enemy counter-chances.",
            "The position remains firmly in hand, showcasing accurate positional judgment.",
        ]
        return f"{openings[h % len(openings)]} {closings[(h >> 4) % len(closings)]}"


# Backward-compatibility alias (strictly routes to Groq API, zero Anthropic calls)
ClaudeTeacherClient = GroqTeacherClient
