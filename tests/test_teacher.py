"""Unit tests for Stage 2: Teacher LLM prompt builder, caching, and client."""

import os
import shutil
import tempfile
import time
import pytest

from chess_commentator.analysis.pipeline import analyze_position
from chess_commentator.teacher.prompt_builder import build_teacher_prompt, TEACHER_SYSTEM_PROMPT
from chess_commentator.teacher.cache import TeacherCache
from chess_commentator.teacher.client import ClaudeTeacherClient
from chess_commentator.teacher.generator import TeacherCommentaryGenerator


@pytest.fixture
def temp_cache_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_build_teacher_prompt():
    fen = "r3k3/8/8/3N4/8/8/8/6K1 w - - 0 1"
    analysis = analyze_position(fen=fen, move="d5c7", depth=8)
    prompt = build_teacher_prompt(analysis)

    assert "Position FEN:" in prompt
    assert "Move Played: Nc7+" in prompt or "c7" in prompt
    assert "Stockfish Evaluation:" in prompt
    assert "Tactical Theme:" in prompt
    assert "Positional Context:" in prompt
    assert len(TEACHER_SYSTEM_PROMPT) > 100


def test_teacher_cache(temp_cache_dir):
    cache = TeacherCache(cache_dir=temp_cache_dir)
    h = cache.compute_hash("fen1", "e2e4", 0, "none", "claude-haiku")
    assert cache.get(h) is None

    cache.set(h, "Solid opening move with e4.")
    assert cache.get(h) == "Solid opening move with e4."
    assert cache.size() == 1


def test_teacher_generator_with_mock_client(temp_cache_dir):
    cache = TeacherCache(cache_dir=temp_cache_dir)
    client = ClaudeTeacherClient(mock_mode=True)
    generator = TeacherCommentaryGenerator(client=client, cache=cache)

    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    analysis = analyze_position(fen=fen, move="Qxf7#", depth=8)

    # First run generates and caches
    comm1, model1, reason1 = generator.generate_for_position(analysis)
    assert len(comm1) > 20
    assert "120b" in model1 or "sonnet" in model1
    assert cache.size() == 1

    # Second run retrieves from cache without calling client again
    comm2, model2, reason2 = generator.generate_for_position(analysis)
    assert comm1 == comm2
    assert "cached" in reason2


def test_route_teacher_model():
    from chess_commentator.teacher.generator import (
        route_teacher_model,
        SONNET_MODEL,
        HAIKU_MODEL,
    )
    from chess_commentator.analysis.models import TaxonomyResult, BoardFeatures

    dummy_features = BoardFeatures(
        phase="middlegame", castled=True, open_files_near_king=0,
        doubled_pawns=0, isolated_pawns=0, passed_pawns=0, mobility=20,
        time_pressure=False, pawns=8, knights=2, bishops=2, rooks=2, queens=1,
        opp_pawns=8, opp_knights=2, opp_bishops=2, opp_rooks=2, opp_queens=1,
        material_balance=0,
    )

    # 1. Blunder with cp_loss >= 300 -> Sonnet
    a_blunder = analyze_position(
        fen="r1bqkbnr/ppp1pppp/3p4/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 3",
        move="f3e5", depth=6,
    )
    model, reason = route_teacher_model(a_blunder)
    assert model == SONNET_MODEL
    assert "300" in reason

    # 2. Checkmate -> Sonnet
    a_mate = analyze_position(
        fen="r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        move="Qxf7#", depth=6,
    )
    model, reason = route_teacher_model(a_mate)
    assert model == SONNET_MODEL
    assert "checkmate" in reason

    # 3. Quiet good move -> Haiku
    a_quiet = analyze_position(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        move="e2e4", depth=6,
    )
    model, reason = route_teacher_model(a_quiet)
    assert model == HAIKU_MODEL
    assert "Standard" in reason


def test_client_fails_loudly_when_credentials_unset(monkeypatch):
    """Ensure GroqTeacherClient raises ValueError when mock_mode=False and GROQ_API_KEY is missing."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("chess_commentator.teacher.client.load_dotenv", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="GROQ_API_KEY is not set and mock_mode=False"):
        ClaudeTeacherClient(mock_mode=False, api_key=None)


def test_rate_limiter():
    """Ensure RateLimiter caps requests and tracks remaining quota accurately."""
    from chess_commentator.teacher.client import RateLimiter
    limiter = RateLimiter(max_requests=5, window_seconds=1.0)
    assert limiter.remaining == 5

    for _ in range(5):
        rem = limiter.acquire()
    assert limiter.remaining == 0

    # 6th request will wait for window to clear
    t0 = time.time()
    limiter.acquire()
    t1 = time.time()
    assert (t1 - t0) >= 0.8  # Ensured backoff wait


def test_spend_report_separates_mock_calls():
    """Ensure mock calls are clearly labeled and separated from real calls in spend summary."""
    client = ClaudeTeacherClient(mock_mode=True)
    comm = client.generate("Move Played: e4 (UCI: e2e4)\nClassification: GOOD", model="qwen/qwen3.8-27b")
    assert len(comm) > 0

    assert len(client.call_history) == 1
    call_record = client.call_history[0]
    assert call_record["is_mock"] is True
    assert "mock (no API call)" in call_record["model"]
    assert call_record["cost_usd"] == 0.0

    summary = client.get_spend_summary()
    assert summary["total_calls"] == 1
    assert summary["real_api_calls"] == 0
    assert summary["mock_calls"] == 1
    assert summary["total_cost_usd"] == 0.0
    assert len(summary["by_model"]) == 0
    assert len(summary["mock_breakdown"]) == 1
    mock_key = list(summary["mock_breakdown"].keys())[0]
    assert "mock (no API call)" in mock_key
    assert summary["mock_breakdown"][mock_key]["calls"] == 1
    assert summary["mock_breakdown"][mock_key]["cost_usd"] == 0.0


def test_teacher_prompt_injection_for_sacrifice_and_zwischenzug():
    """Verify that missed_sacrifice and zwischenzug receive verified continuation injection, while others do not."""
    from chess_commentator.analysis.models import PositionAnalysis, BoardFeatures, TaxonomyResult, MoveMetadata

    fen = "rnbqkbnr/pp3ppp/8/4N3/3p4/2N5/PP1PPPPP/R1BQKB1R w KQkq - 0 6"
    features = BoardFeatures(
        phase="opening",
        castled=False,
        open_files_near_king=0,
        doubled_pawns=0,
        isolated_pawns=0,
        passed_pawns=0,
        mobility=25,
        time_pressure=False,
        pawns=8,
        knights=2,
        bishops=2,
        rooks=2,
        queens=1,
        opp_pawns=7,
        opp_knights=2,
        opp_bishops=2,
        opp_rooks=2,
        opp_queens=1,
        material_balance=78,
    )

    # 1. Zwischenzug -> MUST include continuation injection
    pa_zwischenzug = PositionAnalysis(
        fen=fen,
        move_uci="c3b1",
        move_san="Nb1",
        player_color="white",
        move_number=6,
        eval_before=50,
        eval_after=-300,
        cp_loss=350,
        best_move="d1a4",
        played_best=False,
        mistake_type="blunder",
        taxonomy=TaxonomyResult(mistake_category="missed_tactic", tactic_type="zwischenzug"),
        features=features,
    )
    prompt_zw = build_teacher_prompt(pa_zwischenzug)
    assert "Tactical Continuation:" in prompt_zw
    assert "Engine Best Move: Qa4+" in prompt_zw or "Qa4" in prompt_zw
    assert "Legal Opponent Response (ply+1):" in prompt_zw
    assert "Base your explanation only on the position" in prompt_zw
    assert "CRITICAL INSTRUCTION" not in prompt_zw

    # 2. Missed Sacrifice -> MUST include continuation injection
    pa_sacrifice = PositionAnalysis(
        fen=fen,
        move_uci="c3b1",
        move_san="Nb1",
        player_color="white",
        move_number=6,
        eval_before=50,
        eval_after=-300,
        cp_loss=350,
        best_move="d1a4",
        played_best=False,
        mistake_type="blunder",
        taxonomy=TaxonomyResult(mistake_category="missed_tactic", tactic_type="missed_sacrifice"),
        features=features,
    )
    prompt_sac = build_teacher_prompt(pa_sacrifice)
    assert "Tactical Continuation:" in prompt_sac
    assert "Base your explanation only on the position" in prompt_sac
    assert "CRITICAL INSTRUCTION" not in prompt_sac

    # 3. Missed Mate -> MUST include Forced Mating Line injection
    pa_mate = PositionAnalysis(
        fen="r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 5",
        move_uci="f3e4",
        move_san="Qxe4",
        player_color="white",
        move_number=5,
        eval_before=3000,
        eval_after=100,
        cp_loss=2900,
        best_move="f3f7",
        played_best=False,
        mistake_type="blunder",
        taxonomy=TaxonomyResult(mistake_category="missed_tactic", tactic_type="missed_mate"),
        features=features,
    )
    prompt_mate = build_teacher_prompt(pa_mate)
    assert "Tactical Continuation:" in prompt_mate
    assert "Forced Mating Line" in prompt_mate
    assert "Engine Best Move: Qxf7#" in prompt_mate or "Qxf7" in prompt_mate

    # 4. Checkmate -> MUST include continuation injection
    pa_cm = PositionAnalysis(
        fen="r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 5",
        move_uci="f3f7",
        move_san="Qxf7#",
        player_color="white",
        move_number=5,
        eval_before=3000,
        eval_after=3000,
        cp_loss=0,
        best_move="f3f7",
        played_best=True,
        mistake_type="good_move",
        taxonomy=TaxonomyResult(mistake_category="executed_tactic", tactic_type="checkmate"),
        features=features,
    )
    prompt_cm = build_teacher_prompt(pa_cm)
    assert "OUTCOME: CHECKMATE!" in prompt_cm
    assert "Tactical Continuation:" in prompt_cm

    # 5. Overloaded Piece -> MUST include single ply+1 injection
    pa_overloaded = PositionAnalysis(
        fen=fen,
        move_uci="c3b1",
        move_san="Nb1",
        player_color="white",
        move_number=6,
        eval_before=50,
        eval_after=-300,
        cp_loss=350,
        best_move="d1a4",
        played_best=False,
        mistake_type="blunder",
        taxonomy=TaxonomyResult(mistake_category="missed_tactic", tactic_type="overloaded_piece"),
        features=features,
    )
    prompt_overloaded = build_teacher_prompt(pa_overloaded)
    assert "Tactical Continuation:" in prompt_overloaded
    assert "Legal Opponent Response (ply+1):" in prompt_overloaded

    # 6. Hanging Piece -> MUST include single ply+1 injection
    pa_hanging = PositionAnalysis(
        fen=fen,
        move_uci="c3b1",
        move_san="Nb1",
        player_color="white",
        move_number=6,
        eval_before=50,
        eval_after=-300,
        cp_loss=350,
        best_move="d1a4",
        played_best=False,
        mistake_type="blunder",
        taxonomy=TaxonomyResult(mistake_category="missed_tactic", tactic_type="hanging_piece"),
        features=features,
    )
    prompt_hanging = build_teacher_prompt(pa_hanging)
    assert "Tactical Continuation:" in prompt_hanging
    assert "Legal Opponent Response (ply+1):" in prompt_hanging

    # 7. Fork -> MUST NOT include continuation injection (unscoped)
    pa_fork = PositionAnalysis(
        fen=fen,
        move_uci="c3b1",
        move_san="Nb1",
        player_color="white",
        move_number=6,
        eval_before=50,
        eval_after=-300,
        cp_loss=350,
        best_move="d1a4",
        played_best=False,
        mistake_type="blunder",
        taxonomy=TaxonomyResult(mistake_category="missed_tactic", tactic_type="fork"),
        features=features,
    )
    prompt_fork = build_teacher_prompt(pa_fork)
    assert "Tactical Continuation:" not in prompt_fork


