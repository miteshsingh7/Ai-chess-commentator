import logging
import os
import re
from typing import Optional, Any
from dotenv import load_dotenv
import torch
from chess_commentator.training.model_loader import get_torch_device

load_dotenv()
logger = logging.getLogger(__name__)


class CommentaryInferenceEngine:
    """Manages LLM inference with Groq fast API, optional local LoRA weights, or dynamic fallback."""

    def __init__(
        self,
        base_model_id: str = "microsoft/Phi-3.5-mini-instruct",
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = True,
        device: Optional[str] = None,
        use_groq: bool = True,
        groq_model: str = "qwen/qwen3.8-27b",
    ) -> None:
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.device = torch.device(device) if device else get_torch_device()
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        self.groq_model = groq_model
        self.use_groq = use_groq
        self._groq_client = None

        if self.use_groq:
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key:
                try:
                    import openai
                    self._groq_client = openai.OpenAI(
                        base_url="https://api.groq.com/openai/v1",
                        api_key=api_key,
                    )
                    logger.info("Initialized Groq client with model %s for live commentary.", self.groq_model)
                except Exception as e:
                    logger.warning("Could not initialize Groq client: %s", e)

    def load(self) -> None:
        """Load tokenizer and model with optional adapter onto device if local weights requested."""
        if self._is_loaded:
            return

        # If Groq is already configured and no local adapter path was explicitly provided on disk,
        # skip loading the heavy 7.6GB base model locally to prevent disk-offloading and memory thrash.
        if self._groq_client is not None and not (self.adapter_path and os.path.exists(self.adapter_path)):
            logger.info("Using Groq LLM API for commentary inference (skipping local 3.8B model load).")
            self._is_loaded = True
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id,
                trust_remote_code=False,
            )

            if self.load_in_4bit and torch.cuda.is_available():
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=False,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    device_map="auto" if self.device.type == "mps" else None,
                    torch_dtype=torch.float16,
                    trust_remote_code=False,
                )

            if self.adapter_path:
                is_local = os.path.exists(self.adapter_path)
                is_hub_id = "/" in self.adapter_path and os.sep not in self.adapter_path
                if is_local or is_hub_id:
                    logger.info("Loading LoRA adapter from: %s", self.adapter_path)
                    self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

            self.model.eval()
            self.model.config.use_cache = False
            self._is_loaded = True
        except Exception as exc:
            logger.warning(
                "Local model load skipped or failed: %s (%s). Falling back to Groq / dynamic engine.",
                type(exc).__name__,
                exc,
            )
            self.model = None
            self.tokenizer = None
            self._is_loaded = True

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.3,
    ) -> str:
        """Generate commentary text from formatted prompt."""
        if not self._is_loaded:
            self.load()

        # 1. Local PyTorch model (if loaded and in RAM)
        if self.model is not None and self.tokenizer is not None:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                eos_ids = [self.tokenizer.eos_token_id]
                end_token_id = self.tokenizer.convert_tokens_to_ids("<|end|>")
                if end_token_id is not None and end_token_id not in eos_ids:
                    eos_ids.append(end_token_id)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=(temperature > 0),
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=eos_ids,
                    )
                gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            except Exception as e:
                logger.warning("Local PyTorch generation failed: %s. Trying Groq/dynamic fallback.", e)

        # 2. Groq LLM API (ultra-fast, ~0.5s, no local memory overhead)
        if self._groq_client is not None:
            try:
                groq_out = self._generate_groq(prompt, temperature=temperature)
                if groq_out:
                    return groq_out
            except Exception as e:
                logger.warning("Groq API call failed: %s. Falling back to dynamic heuristic generator.", e)

        # 3. Dynamic heuristic generator grounded in board position features
        return self._heuristic_fallback(prompt)

    def _generate_groq(self, prompt: str, temperature: float = 0.3) -> Optional[str]:
        """Generate high-quality commentary via Groq OpenAI-compatible endpoint."""
        from chess_commentator.dataset.formatter import SYSTEM_INSTRUCTION

        user_content = prompt
        if "<|user|>" in prompt:
            user_content = prompt.split("<|user|>")[-1].split("<|end|>")[0].strip()

        resp = self._groq_client.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_content},
            ],
            max_tokens=180,
            temperature=max(0.1, temperature if temperature > 0 else 0.25),
        )
        content = resp.choices[0].message.content or ""
        return content.strip()

    def _heuristic_fallback(self, prompt: str) -> str:
        """Generate dynamic, feature-grounded commentary when running offline without model."""
        lines = prompt.splitlines()

        # Extract move information
        move_info = next((l for l in lines if l.startswith("Move:")), "Move: played move")
        move_san = move_info.split(":")[1].split("(")[0].strip()

        # Extract evaluation loss and mistake type
        eval_line = next((l for l in lines if "Evaluation before:" in l or "Centipawn loss:" in l), "")
        cp_loss_match = re.search(r'loss:\s*(-?\d+)\s*cp', eval_line)
        cp_loss = int(cp_loss_match.group(1)) if cp_loss_match else 0

        # Extract tactic
        tactic_line = next((l for l in lines if "Tactic:" in l or "Tactical theme:" in l), "")
        tactic = tactic_line.split(":")[-1].strip().lower() if tactic_line else "none"

        # Extract best move
        best_line = next((l for l in lines if "Best move:" in l), "")
        best_move = best_line.split(":")[1].split("(")[0].strip() if best_line else ""

        # Checkmate
        if "#" in move_san or "checkmate" in tactic or "checkmate" in prompt.lower():
            return f"Checkmate! {move_san} delivers the finishing blow, penetrating the opponent's defenses to conclude the game immediately."

        # Blunders with tactical theme
        if "BLUNDER" in prompt or cp_loss >= 300:
            if "knight_fork" in tactic or "fork" in tactic:
                return (
                    f"A catastrophic blunder. {move_san} immediately walks into a crushing knight fork, dropping critical material. "
                    f"Instead of this oversight, {best_move or 'a defensive retreat'} was required to preserve balance."
                )
            if "hanging_piece" in tactic or "hanging" in tactic:
                return (
                    f"A fatal tactical oversight. {move_san} leaves an undefended piece completely hanging for free capture. "
                    f"The initiative swings decisively to the opponent, who can capitalize without delay."
                )
            if "pin" in tactic:
                return (
                    f"A severe miscalculation. {move_san} steps directly into an absolute pin, paralyzing the defender and forfeiting decisive material."
                )
            if "skewer" in tactic:
                return (
                    f"A game-altering blunder. {move_san} exposes high-value pieces to a deadly skewer across the open line, shedding material inevitably."
                )
            if "back_rank" in tactic:
                return (
                    f"Disaster strikes on the back rank. {move_san} fails to create necessary escape breathing room, permitting an irresistible mating assault."
                )
            return (
                f"A costly blunder that concedes {cp_loss} centipawns in a single move. "
                f"{move_san} compromises the position's structural integrity, allowing the opponent to seize a decisive advantage."
            )

        # Mistakes and inaccuracies
        if "MISTAKE" in prompt or cp_loss >= 100:
            return (
                f"{move_san} is an inaccurate continuation that concedes the initiative. "
                f"Rather than consolidating piece harmony with {best_move or 'careful development'}, this move grants the opponent valuable counterplay."
            )

        if cp_loss >= 50:
            return (
                f"A slight inaccuracy. {move_san} releases central tension prematurely. "
                f"While playable, the engine prefers {best_move or 'maintaining piece pressure'} to retain the initiative."
            )

        # Castling
        if move_san in ("O-O", "O-O-O"):
            side = "kingside" if move_san == "O-O" else "queenside"
            return (
                f"With {move_san}, the king is whisked away to safety while the rook activates along the {side} file. "
                f"Securing monarch safety is a textbook fundamental before launching ambitious central maneuvers."
            )

        # Checks
        if "+" in move_san:
            return (
                f"{move_san} delivers an energetic check, seizing the initiative and compelling an immediate defensive response. "
                f"Active forcing moves like this keep the opponent off-balance during the struggle for central control."
            )

        # Captures
        if "x" in move_san:
            return (
                f"With {move_san}, pieces are traded on the board, fundamentally altering the pawn structure and dynamic tension. "
                f"This capture clarifies the position while maintaining optimal piece coordination."
            )

        # Knight moves
        if move_san.startswith("N"):
            return (
                f"Developing the knight with {move_san} increases influence over critical central outposts. "
                f"Active knight placement is essential for coordinating piece play and preparing future tactical thrusts."
            )

        # Bishop moves
        if move_san.startswith("B"):
            return (
                f"{move_san} deploys the bishop along an open diagonal, exerting pressure across the board and facilitating rapid piece coordination."
            )

        # Queen moves
        if move_san.startswith("Q"):
            return (
                f"The queen centralizes with {move_san}, radiating power across multiple ranks and files to spearhead the offensive coordination."
            )

        # Rook moves
        if move_san.startswith("R"):
            return (
                f"Stationing the rook on {move_san} stakes a claim along the file, preparing heavy-piece infiltration into the opponent's territory."
            )

        # Pawn advances
        return (
            f"Advancing with {move_san} stakes an important territorial claim in the center. "
            f"This classical pawn thrust secures spatial superiority while paving pathways for harmonious piece development."
        )

