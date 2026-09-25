"""Unit and integration tests for Stage 6: Live commentary serving."""

import pytest
from chess_commentator.serving.service import (
    generate_commentary,
    construct_serving_prompt,
    CommentaryService,
)
from chess_commentator.analysis.pipeline import analyze_position
from chess_commentator.analysis.models import CommentaryResult


def test_construct_serving_prompt():
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    analysis = analyze_position(fen=fen, move="Qxf7#", depth=8)
    prompt = construct_serving_prompt(analysis)

    assert "<|system|>" in prompt
    assert "<|user|>" in prompt
    assert "<|assistant|>" in prompt
    assert "Qxf7#" in prompt
    assert "Tactical Motif:" in prompt


def test_generate_commentary_checkmate():
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    result: CommentaryResult = generate_commentary(
        fen=fen,
        move="Qxf7#",
        depth=10,
    )

    assert isinstance(result, CommentaryResult)
    assert result.move == "Qxf7#"
    assert result.is_blunder is False
    assert len(result.commentary) > 10
    assert "phase" in result.to_dict()
    assert result.eval_after is not None


def test_generate_commentary_blunder():
    # White hangs knight on e5
    fen = "r1bqkbnr/ppp1pppp/3p4/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 3"
    result: CommentaryResult = generate_commentary(
        fen=fen,
        move="f3e5",
        depth=10,
    )

    assert isinstance(result, CommentaryResult)
    assert result.move == "Ne5"
    assert result.is_blunder is True
    assert result.taxonomy == "hanging_piece"
    assert len(result.commentary) > 10


def test_commentary_service_persistent():
    service = CommentaryService(default_depth=8)
    try:
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        res = service.comment(fen=fen, move="e4")
        assert res.move == "e4"
        assert res.is_blunder is False
    finally:
        service.close()


def test_construct_serving_prompt_continuation_injection():
    # Checkmate move receives tactical continuation injection
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    analysis = analyze_position(fen=fen, move="Qxf7#", depth=8)
    prompt = construct_serving_prompt(analysis)

    assert "Tactical Continuation:" in prompt
    assert "Engine Best Move:" in prompt
    assert "Base your explanation only on the position" in prompt
    assert "CRITICAL INSTRUCTION" not in prompt

    # Quiet move does not receive tactical continuation injection
    fen_quiet = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    analysis_quiet = analyze_position(fen=fen_quiet, move="e4", depth=8)
    prompt_quiet = construct_serving_prompt(analysis_quiet)
    assert "Tactical Continuation:" not in prompt_quiet
    assert "Base your explanation only on the position" not in prompt_quiet


class MockInferenceEngine:
    """Mock inference engine returning scripted completions for testing guardrails."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.temperatures_used = []

    def generate(self, prompt: str, temperature: float = 0.0, **kwargs) -> str:
        self.temperatures_used.append(temperature)
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
        else:
            resp = self.responses[-1] if self.responses else "Default completion."
        self.call_count += 1
        return resp

    def _heuristic_fallback(self, prompt: str) -> str:
        return "Grounded heuristic fallback template response."


def test_guardrail_passes_valid_completion_unchanged():
    valid_text = "A principled, confident choice. e4 reinforces central stability, active piece coordination, and fluid development."
    mock_inf = MockInferenceEngine(responses=[valid_text])
    service = CommentaryService(inference_engine=mock_inf, default_depth=8)
    try:
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        res = service.comment(fen=fen, move="e4")

        assert res.commentary == valid_text
        assert service.first_pass_rejection_count == 0
        assert service.total_generations_count == 1
        assert mock_inf.call_count == 1
    finally:
        service.close()


def test_guardrail_triggers_fallback_on_grounding_failure():
    # Model produces illegal moves / non-existent pieces on board
    hallucination = "White plays 1. Qe8 and captures the knight on e5."
    mock_inf = MockInferenceEngine(responses=[hallucination, hallucination])
    service = CommentaryService(inference_engine=mock_inf, default_depth=8)
    try:
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        res = service.comment(fen=fen, move="e4")

        assert res.commentary == "Grounded heuristic fallback template response."
        assert service.first_pass_rejection_count == 1
        assert service.total_generations_count == 1
        # Called twice: once at temp=0.0, once regeneration at temp=0.3
        assert mock_inf.call_count == 2
        assert mock_inf.temperatures_used == [0.0, 0.3]
    finally:
        service.close()


def test_guardrail_regeneration_success():
    # First pass hallucinates, but second pass (temperature=0.3) is grounded
    hallucination = "White plays 1. Qe8 and captures the knight on e5."
    valid_text = "A principled, confident choice. e4 reinforces central stability, active piece coordination, and fluid development."
    mock_inf = MockInferenceEngine(responses=[hallucination, valid_text])
    service = CommentaryService(inference_engine=mock_inf, default_depth=8)
    try:
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        res = service.comment(fen=fen, move="e4")

        assert res.commentary == valid_text
        assert service.first_pass_rejection_count == 1
        assert service.total_generations_count == 1
        assert mock_inf.call_count == 2
        assert mock_inf.temperatures_used == [0.0, 0.3]
    finally:
        service.close()


def test_guardrail_eval_sign_contradiction_triggers_fallback():
    # Blunder move described with praise word
    blunder_praise = "A brilliant masterpiece! White executes a flawless, fantastic sacrifice on e5."
    mock_inf = MockInferenceEngine(responses=[blunder_praise, blunder_praise])
    service = CommentaryService(inference_engine=mock_inf, default_depth=8)
    try:
        # Hanging knight move
        fen = "r1bqkbnr/ppp1pppp/3p4/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 3"
        res = service.comment(fen=fen, move="f3e5")

        assert res.commentary == "Grounded heuristic fallback template response."
        assert service.first_pass_rejection_count == 1
        assert mock_inf.call_count == 2
    finally:
        service.close()

