"""Script to run pilot generation on the 50 stratified positions with Sonnet/Haiku routing."""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from chess_commentator.analysis.models import PositionAnalysis, BoardFeatures, TaxonomyResult
from chess_commentator.teacher.client import ClaudeTeacherClient
from chess_commentator.teacher.cache import TeacherCache
from chess_commentator.teacher.generator import (
    TeacherCommentaryGenerator,
    route_teacher_model,
    SONNET_MODEL,
    HAIKU_MODEL,
)


def row_to_analysis(r: dict) -> PositionAnalysis:
    features = BoardFeatures(
        phase=r.get("phase", "middlegame"),
        castled=bool(r.get("castled", False)),
        open_files_near_king=int(r.get("open_files_near_king", 0)),
        doubled_pawns=int(r.get("doubled_pawns", 0)),
        isolated_pawns=int(r.get("isolated_pawns", 0)),
        passed_pawns=int(r.get("passed_pawns", 0)),
        mobility=int(r.get("mobility", 20)),
        time_pressure=bool(r.get("time_pressure", False)),
        pawns=int(r.get("pawns", 8)),
        knights=int(r.get("knights", 2)),
        bishops=int(r.get("bishops", 2)),
        rooks=int(r.get("rooks", 2)),
        queens=int(r.get("queens", 1)),
        opp_pawns=int(r.get("opp_pawns", 8)),
        opp_knights=int(r.get("opp_knights", 2)),
        opp_bishops=int(r.get("opp_bishops", 2)),
        opp_rooks=int(r.get("opp_rooks", 2)),
        opp_queens=int(r.get("opp_queens", 1)),
        material_balance=int(r.get("material_balance", 0)),
    )
    taxonomy = TaxonomyResult(
        mistake_category=r.get("mistake_category", "none"),
        tactic_type=r.get("tactic_type", "none"),
        piece_lost=r.get("piece_lost", "none"),
        mate_missed=int(r.get("mate_missed", 0)),
    )
    return PositionAnalysis(
        fen=r["fen"],
        move_uci=r["move_uci"],
        move_san=r.get("move_san", r["move_uci"]),
        player_color=r.get("player_color", "white"),
        move_number=int(r.get("move_number", 20)),
        eval_before=r.get("eval_before"),
        eval_after=r.get("eval_after"),
        cp_loss=r.get("cp_loss"),
        best_move=r.get("best_move"),
        played_best=bool(r.get("played_best", False)),
        mistake_type=r.get("mistake_type", "good"),
        taxonomy=taxonomy,
        features=features,
    )


import argparse


def run_pilot(mock: bool = False) -> pd.DataFrame:
    pilot_path = "data/pilot_50_positions.parquet"
    if not os.path.exists(pilot_path):
        raise FileNotFoundError(f"{pilot_path} not found. Run scripts/sample_pilot_50.py first.")

    df = pd.read_parquet(pilot_path)
    print(f"Loaded {len(df)} stratified pilot positions.")

    api_key = os.environ.get("GROQ_API_KEY")
    if not mock:
        if not api_key:
            raise ValueError(
                "❌ GROQ_API_KEY is not set in .env or the environment!\n"
                "Real Groq API generation cannot proceed without valid credentials.\n"
                "To run a real generation, set GROQ_API_KEY in .env or export GROQ_API_KEY='gsk_...'.\n"
                "If you explicitly intend to run an offline mock simulation, pass the --mock flag."
            )
        print("GROQ_API_KEY is present. Executing REAL Groq API generation.")
    else:
        print("⚠️  Running in explicit MOCK mode (--mock flag provided). No real API calls will be made.")

    client = ClaudeTeacherClient(mock_mode=mock)
    cache = TeacherCache(cache_dir=".cache/pilot_commentary")
    generator = TeacherCommentaryGenerator(client=client, cache=cache, enable_routing=True)

    analyses = [row_to_analysis(r) for r in df.to_dict("records")]
    res_df = generator.generate_batch(analyses, output_parquet="data/pilot_50_with_commentary.parquet")

    spend_summary = client.get_spend_summary()
    print("\n" + "=" * 65)
    print("💰 GROQ PILOT GENERATION USAGE REPORT")
    print("=" * 65)
    print(f"Real API Calls Made: {spend_summary['real_api_calls']} / {spend_summary['total_calls']}")
    print(f"Total Input Tokens:  {spend_summary['total_input_tokens']}")
    print(f"Total Output Tokens: {spend_summary['total_output_tokens']}")
    print(f"Total Financial Spend: ${spend_summary['total_cost_usd']:.4f} USD (Groq Free Tier)")

    if spend_summary["by_model"]:
        print("\nReal Groq API Calls by Model:")
        for m, d in spend_summary["by_model"].items():
            print(f"  • {m}: {d['calls']} calls, {d['in_tokens']} in, {d['out_tokens']} out, ${d['cost_usd']:.4f} USD")

    if spend_summary["mock_breakdown"]:
        print("\nMock / Heuristic Generations (no real API calls):")
        for m, d in spend_summary["mock_breakdown"].items():
            print(f"  • {m}: {d['calls']} mock calls, $0.0000 USD (simulated tokens: {d['simulated_in_tokens']} in, {d['simulated_out_tokens']} out)")
    print("=" * 65 + "\n")

    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pilot teacher commentary generation for 50 positions.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Explicitly enable mock mode (offline/testing with zero API calls).",
    )
    args = parser.parse_args()
    run_pilot(mock=args.mock)
