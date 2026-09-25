"""Local verification that the smoke test evaluation pipeline runs without errors."""

import json
from chess_commentator.dataset.checker import validate_chess_grounding
from chess_commentator.dataset.filters import validate_no_scaffolding_leakage

def verify():
    with open("data/smoke_test_v8_inputs.json") as f:
        inputs = json.load(f)

    print(f"Loaded {len(inputs)} smoke test positions.")
    for p in inputs:
        cat = p["category"]
        san = p["move_san"]
        uci = p["move_uci"]
        fen = p["fen"]
        best_uci = p.get("best_move")

        # Test dummy text
        dummy_comm = f"The move {san} is played to control key squares and develop pieces."
        chk_g = validate_chess_grounding(fen, uci, dummy_comm, best_uci)
        chk_l = validate_no_scaffolding_leakage(dummy_comm)

        assert chk_l.passed is True
        print(f"  [{cat}] {san} ({uci}): Grounding check returned {chk_g.passed}, Leakage check returned {chk_l.passed}")

    print("\n✓ Smoke test evaluation harness verified successfully!")

if __name__ == "__main__":
    verify()
