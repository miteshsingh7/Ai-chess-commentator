"""Interactive terminal CLI for generating commentary on arbitrary positions and moves."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from chess_commentator.serving.service import generate_commentary


def main() -> None:
    parser = argparse.ArgumentParser(description="Live AI Chess Commentator CLI")
    parser.add_argument(
        "--fen",
        type=str,
        default="r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        help="FEN board position string",
    )
    parser.add_argument(
        "--move",
        type=str,
        default="Qxf7#",
        help="Played move in UCI or SAN notation (e.g. 'e4' or 'Qxf7#' or 'e2e4')",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=18,
        help="Stockfish evaluation depth (default: 18)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to trained QLoRA adapter directory",
    )
    args = parser.parse_args()

    print("\n♟️  AI CHESS COMMENTATOR — LIVE SERVING")
    print("=" * 55)
    print(f"FEN:   {args.fen}")
    print(f"Move:  {args.move}")
    print(f"Depth: {args.depth}")
    print("Analyzing position with Stockfish & classifying taxonomy...")

    try:
        result = generate_commentary(
            fen=args.fen,
            move=args.move,
            depth=args.depth,
            adapter_path=args.adapter,
        )

        print("\n" + "=" * 55)
        print("🎯 ANALYSIS & COMMENTARY RESULT:")
        print("=" * 55)
        print(f"Played Move:     {result.move}")
        print(f"Game Phase:      {result.phase.title()}")
        print(f"Centipawn Loss:  {result.cp_loss} cp")
        print(f"Evaluation:      {result.eval_after:+d} cp")
        print(f"Top Engine Move: {result.best_move}")
        print(f"Tactical Motif:  {result.taxonomy}")
        print(f"Blunder Flag:    {'YES' if result.is_blunder else 'NO'}")
        print("-" * 55)
        print("🎙️ COMMENTARY:")
        print(f"\"{result.commentary}\"")
        print("=" * 55 + "\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
