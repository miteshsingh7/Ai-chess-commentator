"""CLI to run domain analysis and taxonomy classification on PGN files."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from chess_commentator.analysis.pipeline import analyze_pgn_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PGN game files with Stockfish and taxonomy.")
    parser.add_argument("--pgn", type=str, required=True, help="Path to input PGN file.")
    parser.add_argument("--output", type=str, default="data/processed/moves_analyzed.parquet", help="Output parquet path.")
    parser.add_argument("--player", type=str, default=None, help="Optional username to filter moves.")
    parser.add_argument("--depth", type=int, default=18, help="Stockfish search depth (default: 18).")
    parser.add_argument("--mode", type=str, default="deep", choices=["fast", "deep"], help="Analysis mode.")
    args = parser.parse_args()

    print(f"Analyzing PGN '{args.pgn}' (mode: {args.mode}, depth: {args.depth})...")
    try:
        df = analyze_pgn_file(
            pgn_path=args.pgn,
            output_parquet=args.output,
            player_username=args.player,
            depth=args.depth,
            mode=args.mode,
        )
        print(f"Successfully analyzed {len(df)} moves. Saved to: {args.output}")
    except Exception as e:
        print(f"Analysis failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
