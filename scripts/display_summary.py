"""Display Stage 5 summary metrics formatted cleanly."""

import json
from pathlib import Path


def main() -> None:
    path = Path(".stage5_output/stage5_summary.json")
    if not path.exists():
        print(f"File not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    overall = data["overall"]
    cats = data["per_category"]

    print("=" * 115)
    print(f"{'CATEGORY':<25} | {'N':<4} | {'ROUGE-L':<8} | {'F1':<6} | {'SENTIMENT':<16} | {'TACTIC REC':<16} | {'GROUNDING':<20} | {'LEAKS':<7}")
    print("-" * 115)

    for m, c in cats.items():
        n = c["n"]
        rl = f"{c['rouge_l']:.4f}"
        f1 = f"{c['token_f1']:.4f}"
        sent = f"{c['eval_sentiment_passed']}/{n} ({c['eval_sentiment_rate']*100:4.1f}%)"
        if c["tactic_recalled"] is not None:
            trec = f"{c['tactic_recalled']}/{n} ({c['tactic_recall_rate']*100:4.1f}%)"
        else:
            trec = "N/A"
        grnd = f"{c['grounding_passed']}/{n} ({c['grounding_rate']*100:4.1f}%)"
        leaks = f"{c['scaffolding_leaks']}/{n}"
        print(f"{m:<25} | {n:^4} | {rl:^8} | {f1:^6} | {sent:<16} | {trec:<16} | {grnd:<20} | {leaks:<7}")

    print("=" * 115)
    print("OVERALL METRICS (N=292):")
    print(f"  ROUGE-L:                 {overall['rouge_l']:.4f}")
    print(f"  Token-F1:                {overall['token_f1']:.4f}")
    print(f"  Eval-Sentiment Agree:    {overall['eval_sentiment_agreement']*100:.2f}%")
    print(f"  Tactic Recall:           {overall['tactic_recall']*100:.2f}%")
    print(f"  Chess Grounding Pass:    {overall['chess_grounding_pass_rate']*100:.2f}%")
    print(f"  Hallucination Rate:      {overall['hallucination_rate']*100:.2f}%")
    print(f"  Scaffolding Leaks:       {overall['scaffolding_leakage_count']}/292 ({overall['scaffolding_leakage_rate']*100:.2f}%)")
    print("=" * 115)


if __name__ == "__main__":
    main()
