import sys
import pandas as pd

df = pd.read_parquet("data/pilot_50_with_commentary.parquet")
start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
end = int(sys.argv[2]) if len(sys.argv) > 2 else len(df)
print(f"Printing records {start} to {end} of {len(df)}:\n")

for i, r in enumerate(df.to_dict("records")[start - 1:end], start):
    fen = r["fen"]
    move = r.get("move_san", r["move_uci"])
    loss = r.get("cp_loss", 0)
    mtype = r.get("mistake_type", "good")
    tactic = r.get("tactic_type", "none")
    raw_model = r.get("teacher_model", "")
    if "sonnet" in raw_model:
        model_display = "Claude 3.5 Sonnet"
    elif "haiku" in raw_model:
        model_display = "Claude 3.5 Haiku"
    else:
        model_display = raw_model
    reason = r.get("teacher_routing_reason", "")
    comm = r.get("teacher_commentary", "")

    print(f"[{i:02d}] Move: {move} | FEN: {fen}")
    print(f"     Eval: {mtype.upper()} ({loss} cp loss) | Motif: {tactic}")
    print(f"     Routed Model: {model_display} — Reason: {reason}")
    print(f"     Commentary: \"{comm}\"")
    print("-" * 75)
