"""Parse and display full training metrics and loss curves."""

import json

def parse_metrics():
    state_path = ".kaggle_output_v9/phi35_chess_commentator_adapter/checkpoint-366/trainer_state.json"
    with open(state_path) as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    print("=" * 65)
    print("STAGE 4 QLORA TRAINING METRICS (Phi-3.5-mini-instruct)")
    print("=" * 65)
    print(f"Total Steps: {state.get('global_step', 366)} / {state.get('max_steps', 366)} (3.0 epochs)")
    print(f"Total FLOPs: {state.get('total_flos')}")
    print("-" * 65)
    print(f"{'Step':>6} | {'Epoch':>6} | {'Train Loss':>12} | {'Eval Loss':>12} | {'Grad Norm':>10}")
    print("-" * 65)

    for entry in log_history:
        step = entry.get("step")
        epoch = entry.get("epoch", 0.0)
        train_loss = entry.get("loss")
        eval_loss = entry.get("eval_loss")
        grad_norm = entry.get("grad_norm")

        tr_str = f"{train_loss:12.4f}" if train_loss is not None else f"{'--':>12}"
        ev_str = f"{eval_loss:12.4f}" if eval_loss is not None else f"{'--':>12}"
        gn_str = f"{grad_norm:10.4f}" if grad_norm is not None else f"{'--':>10}"

        print(f"{step:6d} | {epoch:6.2f} | {tr_str} | {ev_str} | {gn_str}")

    print("=" * 65)

if __name__ == "__main__":
    parse_metrics()
