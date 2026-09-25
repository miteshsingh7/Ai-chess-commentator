"""Display per-category LLM Judge breakdown for real evaluations."""

import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    path = Path("data/stage5_judge_scores.json")
    if not path.exists():
        print(f"File not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    real = [x for x in data if "Clear explanation with accurate tactical grounding" not in x["judge_score"]["feedback"]]
    by_cat = defaultdict(list)
    for x in real:
        by_cat[x["motif"]].append(x["judge_score"])

    print("=" * 75)
    print(f"{'CATEGORY':<25} | {'N (real)':<8} | {'ACC':<5} | {'EDU':<5} | {'NAT':<5} | {'CONC':<5} | {'OVERALL':<7}")
    print("-" * 75)

    for m, scores in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        n = len(scores)
        acc = sum(s["accuracy"] for s in scores) / n
        edu = sum(s["educational_value"] for s in scores) / n
        nat = sum(s["naturalness"] for s in scores) / n
        conc = sum(s["conciseness"] for s in scores) / n
        ovr = sum(s["overall"] for s in scores) / n
        print(f"{m:<25} | {n:^8} | {acc:^5.2f} | {edu:^5.2f} | {nat:^5.2f} | {conc:^5.2f} | {ovr:^7.2f}")

    print("=" * 75)
    total = len(real)
    m_acc = sum(s["accuracy"] for scores in by_cat.values() for s in scores) / total
    m_edu = sum(s["educational_value"] for scores in by_cat.values() for s in scores) / total
    m_nat = sum(s["naturalness"] for scores in by_cat.values() for s in scores) / total
    m_conc = sum(s["conciseness"] for scores in by_cat.values() for s in scores) / total
    m_ovr = sum(s["overall"] for scores in by_cat.values() for s in scores) / total
    print(f"{'TOTAL REAL EVALS':<25} | {total:^8} | {m_acc:^5.2f} | {m_edu:^5.2f} | {m_nat:^5.2f} | {m_conc:^5.2f} | {m_ovr:^7.2f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
