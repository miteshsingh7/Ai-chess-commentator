"""Stockfish engine interface for evaluation and centipawn loss calculation.

Provides cross-platform binary detection, fast/deep analysis, and FEN caching.
"""

import os
import shutil
import sys
from typing import Optional, Tuple, Dict, Any
import chess
import chess.engine

from chess_commentator.analysis.constants import (
    PIECE_VALUE,
    CP_CAP,
    GOOD_THRESHOLD,
    INACCURACY_THRESHOLD,
    MISTAKE_THRESHOLD,
)


def find_stockfish() -> str:
    """Auto-detect Stockfish binary path across macOS, Linux, and Windows."""
    # 1. Environment variable
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. On PATH
    sf_which = shutil.which("stockfish")
    if sf_which:
        return sf_which

    # 3. Known system locations
    candidates = []
    if sys.platform == "darwin":
        candidates = [
            "/opt/homebrew/bin/stockfish",
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
        ]
    elif sys.platform == "win32":
        candidates = [
            r"C:\stockfish\stockfish.exe",
            r"C:\Program Files\Stockfish\stockfish.exe",
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "stockfish", "stockfish.exe"),
        ]
    else:  # Linux
        candidates = [
            "/usr/games/stockfish",
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
        ]

    for cand in candidates:
        if os.path.isfile(cand):
            return cand

    return "stockfish"


def stockfish_to_cp(score: chess.engine.PovScore) -> int:
    """Convert chess.engine.PovScore to white-perspective centipawns capped at CP_CAP."""
    white_score = score.white()
    if white_score.is_mate():
        mate_moves = white_score.mate()
        if mate_moves is not None:
            return CP_CAP if mate_moves > 0 else -CP_CAP
        return CP_CAP
    raw_cp = white_score.score()
    if raw_cp is None:
        return 0
    return max(-CP_CAP, min(CP_CAP, raw_cp))


def compute_cp_loss(
    white_before: Optional[int],
    white_after: Optional[int],
    color: str,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Compute player centipawn loss and perspective evals from white scores."""
    if white_before is None or white_after is None:
        return None, None, None

    if color.lower() == "white":
        cp_loss = white_before - white_after
        eval_before = white_before
        eval_after = white_after
    else:
        cp_loss = white_after - white_before
        eval_before = -white_before
        eval_after = -white_after

    return cp_loss, eval_before, eval_after


def classify_mistake(cp_loss: Optional[int]) -> str:
    """Classify centipawn loss into standard qualitative mistake category."""
    if cp_loss is None:
        return "unknown"
    if cp_loss < GOOD_THRESHOLD:
        return "good"
    if cp_loss < INACCURACY_THRESHOLD:
        return "inaccuracy"
    if cp_loss < MISTAKE_THRESHOLD:
        return "mistake"
    return "blunder"


def is_suspicious(board: chess.Board, move: chess.Move) -> bool:
    """Fast-mode heuristic: detect if a move is potentially tactically suspect."""
    try:
        color = board.turn
        opp = not color

        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if captured and attacker:
                cap_val = PIECE_VALUE.get(captured.piece_type, 0)
                att_val = PIECE_VALUE.get(attacker.piece_type, 0)
                b2 = board.copy()
                b2.push(move)
                if att_val > cap_val and b2.attackers(opp, move.to_square):
                    return True
                if att_val > cap_val + 1:
                    return True
            if captured and not board.attackers(opp, move.to_square):
                return False

        b2 = board.copy()
        b2.push(move)
        if b2.is_check():
            return True

        for piece_type in (chess.QUEEN, chess.ROOK):
            for sq in b2.pieces(piece_type, color):
                if b2.attackers(opp, sq) and not b2.attackers(color, sq):
                    return True

        for piece_type in (chess.BISHOP, chess.KNIGHT):
            for sq in b2.pieces(piece_type, color):
                if b2.attackers(opp, sq) and not b2.attackers(color, sq):
                    piece = b2.piece_at(sq)
                    if piece:
                        piece_val = PIECE_VALUE.get(piece.piece_type, 0)
                        for att_sq in b2.attackers(opp, sq):
                            att = b2.piece_at(att_sq)
                            if att and PIECE_VALUE.get(att.piece_type, 9) < piece_val:
                                return True
    except Exception:
        return True
    return False


class StockfishEngine:
    """Wrapper managing the Stockfish process and caching evaluations."""

    def __init__(
        self,
        executable_path: Optional[str] = None,
        threads: int = 2,
        hash_mb: int = 128,
        default_depth: int = 18,
    ) -> None:
        self.executable_path = executable_path or find_stockfish()
        self.threads = threads
        self.hash_mb = hash_mb
        self.default_depth = default_depth
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._cache: Dict[str, Dict[str, Any]] = {}

    def __enter__(self) -> "StockfishEngine":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def start(self) -> None:
        """Start the UCI engine subprocess."""
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.executable_path)
            self._engine.configure({"Threads": self.threads, "Hash": self.hash_mb})

    def close(self) -> None:
        """Terminate the UCI engine subprocess."""
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    def evaluate_board(
        self,
        board: chess.Board,
        depth: Optional[int] = None,
        time_limit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single board state, returning centipawns and best move."""
        fen = board.fen()
        if fen in self._cache:
            return self._cache[fen]

        if board.is_checkmate():
            score_cp = -CP_CAP if board.turn == chess.WHITE else CP_CAP
            result = {"score_cp": score_cp, "best_move": None}
            self._cache[fen] = result
            return result

        if board.is_stalemate() or board.is_insufficient_material():
            result = {"score_cp": 0, "best_move": None}
            self._cache[fen] = result
            return result

        if self._engine is None:
            self.start()

        assert self._engine is not None

        if time_limit is not None:
            limit = chess.engine.Limit(time=time_limit)
        else:
            limit = chess.engine.Limit(depth=depth or self.default_depth)

        info = self._engine.analyse(board, limit)
        score_cp = stockfish_to_cp(info["score"])
        pv = info.get("pv", [])
        best_move_uci = pv[0].uci() if pv else None

        result = {
            "score_cp": score_cp,
            "best_move": best_move_uci,
        }
        self._cache[fen] = result
        return result

    def evaluate_move(
        self,
        board: chess.Board,
        move: chess.Move,
        depth: Optional[int] = None,
        time_limit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate a move: score before, score after, centipawn loss, and best move."""
        color_str = "white" if board.turn == chess.WHITE else "black"
        before_data = self.evaluate_board(board, depth=depth, time_limit=time_limit)

        b_after = board.copy()
        b_after.push(move)
        after_data = self.evaluate_board(b_after, depth=depth, time_limit=time_limit)

        white_before = before_data["score_cp"]
        white_after = after_data["score_cp"]
        best_uci = before_data["best_move"]

        cp_loss, eval_before, eval_after = compute_cp_loss(white_before, white_after, color_str)
        played_best = (best_uci == move.uci()) if best_uci else False

        if played_best and cp_loss is not None and cp_loss > 0:
            cp_loss = 0

        return {
            "eval_before": eval_before,
            "eval_after": eval_after,
            "cp_loss": cp_loss,
            "best_move": best_uci,
            "played_best": played_best,
            "mistake_type": "good" if played_best else classify_mistake(cp_loss),
        }
