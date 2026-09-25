"""CLI to generate teacher commentary on analyzed positions using Groq API with caching."""

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import pandas as pd
from chess_commentator.analysis.models import PositionAnalysis, BoardFeatures, TaxonomyResult
from chess_commentator.teacher.client import GroqTeacherClient
from chess_commentator.teacher.cache import TeacherCache
from chess_commentator.teacher.generator import TeacherCommentaryGenerator


def row_to_analysis(row: dict) -> PositionAnalysis:
    features = BoardFeatures(
        phase=row.get("phase", "middlegame"),
        castled=bool(row.get("castled", False)),
        open_files_near_king=int(row.get("open_files_near_king", 0)),
        doubled_pawns=int(row.get("doubled_pawns", 0)),
        isolated_pawns=int(row.get("isolated_pawns", 0)),
        passed_pawns=int(row.get("passed_pawns", 0)),
        mobility=int(row.get("mobility", 20)),
        time_pressure=bool(row.get("time_pressure", False)),
        pawns=int(row.get("pawns", 8)),
        knights=int(row.get("knights", 2)),
        bishops=int(row.get("bishops", 2)),
        rooks=int(row.get("rooks", 2)),
        queens=int(row.get("queens", 1)),
        opp_pawns=int(row.get("opp_pawns", 8)),
        opp_knights=int(row.get("opp_knights", 2)),
        opp_bishops=int(row.get("opp_bishops", 2)),
        opp_rooks=int(row.get("opp_rooks", 2)),
        opp_queens=int(row.get("opp_queens", 1)),
        material_balance=int(row.get("material_balance", 0)),
    )
    taxonomy = TaxonomyResult(
        mistake_category=row.get("mistake_category", "none"),
        tactic_type=row.get("tactic_type", "none"),
        piece_lost=row.get("piece_lost", "none"),
        mate_missed=int(row.get("mate_missed", 0)),
    )
    return PositionAnalysis(
        fen=row["fen"],
        move_uci=row["move_uci"],
        move_san=row.get("move_san", row["move_uci"]),
        player_color=row.get("player_color", "white"),
        move_number=int(row.get("move_number", 20)),
        eval_before=row.get("eval_before"),
        eval_after=row.get("eval_after"),
        cp_loss=row.get("cp_loss"),
        best_move=row.get("best_move"),
        played_best=bool(row.get("played_best", False)),
        mistake_type=row.get("mistake_type", "good"),
        taxonomy=taxonomy,
        features=features,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic teacher commentary for chess positions.")
    parser.add_argument("--input", type=str, required=True, help="Input Parquet file with analyzed positions.")
    parser.add_argument("--output", type=str, default="data/processed/moves_with_commentary.parquet", help="Output Parquet path.")
    parser.add_argument("--model", type=str, default="qwen/qwen3.8-27b", help="Groq model ID.")
    parser.add_argument("--max-samples", type=int, default=None, help="Max positions to process.")
    parser.add_argument("--cache-dir", type=str, default=".cache/teacher_commentary", help="Cache directory.")
    parser.add_argument("--mock", action="store_true", help="Run with mock teacher client (no API calls).")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    if args.max_samples:
        df = df.iloc[:args.max_samples]

    api_key = os.environ.get("GROQ_API_KEY")
    if not args.mock:
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

    print(f"Loaded {len(df)} positions. Model: {args.model}, Mock mode: {args.mock}")
    client = GroqTeacherClient(model=args.model, mock_mode=args.mock)
    cache = TeacherCache(cache_dir=args.cache_dir)
    generator = TeacherCommentaryGenerator(client=client, cache=cache)

    analyses = [row_to_analysis(r) for r in df.to_dict("records")]
    res_df = generator.generate_batch(analyses, output_parquet=args.output)
    print(f"Generation complete! Output saved to: {args.output}")

    spend_summary = client.get_spend_summary()
    print("\n" + "=" * 65)
    print("💰 TEACHER GENERATION SPEND REPORT")
    print("=" * 65)
    print(f"Real API Calls Made: {spend_summary['real_api_calls']} / {spend_summary['total_calls']}")
    print(f"Total Input Tokens:  {spend_summary['total_input_tokens']}")
    print(f"Total Output Tokens: {spend_summary['total_output_tokens']}")
    print(f"Total Financial Spend: ${spend_summary['total_cost_usd']:.4f} USD")

    if spend_summary["by_model"]:
        print("\nReal Groq API Calls by Model:")
        for m, d in spend_summary["by_model"].items():
            print(f"  • {m}: {d['calls']} calls, {d['in_tokens']} in, {d['out_tokens']} out, ${d['cost_usd']:.4f} USD")

    if spend_summary["mock_breakdown"]:
        print("\nMock / Heuristic Generations (no real API calls):")
        for m, d in spend_summary["mock_breakdown"].items():
            print(f"  • {m}: {d['calls']} mock calls, $0.0000 USD (simulated tokens: {d['simulated_in_tokens']} in, {d['simulated_out_tokens']} out)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
