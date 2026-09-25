"""Evaluate smoke test results jointly for scaffolding leakage and chess grounding."""

import json
import os
import pandas as pd
from chess_commentator.dataset.checker import validate_chess_grounding
from chess_commentator.dataset.filters import validate_no_scaffolding_leakage

def evaluate():
    results_path = ".smoke_test_output_v8/smoke_test_results.json"
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing {results_path}")

    with open(results_path) as f:
        results = json.load(f)

    eval_results = []
    print("=" * 80)
    print("JOINT SMOKE TEST EVALUATION: SCAFFOLDING LEAKAGE & CHESS GROUNDING")
    print("=" * 80)

    leakage_passes = 0
    grounding_passes = 0

    for idx, p in enumerate(results, start=1):
        cat = p["category"]
        fen = p["fen"]
        san = p["move_san"]
        uci = p["move_uci"]
        color = p["color"]
        engine_eval = p["engine"]
        comm = p["generated_commentary"]
        best_uci = p.get("best_move")

        chk_leak = validate_no_scaffolding_leakage(comm)
        chk_ground = validate_chess_grounding(
            fen=fen,
            move_uci=uci,
            commentary=comm,
            best_move_uci=best_uci,
        )

        has_leak = not chk_leak.passed
        passed_grounding = chk_ground.passed

        if not has_leak:
            leakage_passes += 1
        if passed_grounding:
            grounding_passes += 1

        eval_results.append({
            "index": idx,
            "category": cat,
            "fen": fen,
            "move_san": san,
            "move_uci": uci,
            "color": color,
            "engine": engine_eval,
            "best_move": best_uci,
            "generated_commentary": comm,
            "scaffolding_leak": has_leak,
            "leakage_reason": chk_leak.reason,
            "grounding_passed": passed_grounding,
            "grounding_reason": chk_ground.reason,
        })

        print(f"\n--- Position {idx}: [{cat.upper()}] {san} ({uci}) by {color} ---")
        print(f"Engine: {engine_eval}")
        print(f"Generated Commentary:\n  \"{comm}\"")
        print(f"Scaffolding Leakage: {'NO (CLEAN)' if not has_leak else 'YES (LEAK: ' + chk_leak.reason + ')'}")
        print(f"Chess Grounding:     {'PASS' if passed_grounding else 'FAIL (' + chk_ground.reason + ')'}")

    # Summary Table
    df = pd.DataFrame([
        {
            "Pos": r["index"],
            "Category": r["category"],
            "Move": r["move_san"],
            "Scaffolding Leak?": "NO" if not r["scaffolding_leak"] else "YES",
            "Grounding Passed?": "PASS" if r["grounding_passed"] else "FAIL",
            "Grounding Reason": r["grounding_reason"],
        }
        for r in eval_results
    ])

    print("\n" + "=" * 80)
    print("FINAL JOINT VALIDATION SCORECARD")
    print("=" * 80)
    print(df.to_string(index=False))

    print("-" * 80)
    print(f"Total Scaffolding Leaks: {8 - leakage_passes} / 8 (Requirement: 0 / 8, {'PASSED' if leakage_passes == 8 else 'FAILED'})")
    print(f"Total Grounding Passes:  {grounding_passes} / 8 (Requirement: >= 6 / 8, {'PASSED' if grounding_passes >= 6 else 'FAILED'})")
    print("=" * 80)

    out_path = "data/smoke_test_v8_final_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved detailed evaluation to {out_path}")

if __name__ == "__main__":
    evaluate()
