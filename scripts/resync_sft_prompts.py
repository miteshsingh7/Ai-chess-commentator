"""Resynchronize SFT dataset prompts with aligned Tactical Continuation blocks and natural constraints.
Zero Groq spend: relies entirely on existing curated commentary and local Stockfish.
"""

import json
import os
import sys
import time
from typing import Dict, Any, List
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "src")

from chess_commentator.analysis.engine import StockfishEngine
from chess_commentator.dataset.formatter import (
    SYSTEM_INSTRUCTION,
    construct_user_prompt,
)
from chess_commentator.teacher.prompt_builder import CONTINUATION_TACTIC_TYPES


def parse_fen_and_uci_from_user_text(user_text: str) -> tuple[str, str]:
    """Extract fen and move_uci from existing user turn text."""
    lines = user_text.strip().split("\n")
    fen = lines[0].replace("Position: ", "").strip()
    move_line = lines[1]
    uci = move_line.split("(")[1].split(")")[0].strip()
    return fen, uci


def resync_splits():
    parquet_path = "data/full_3000_clean_curated.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Missing {parquet_path}")

    print(f"Loading curated positions from {parquet_path}...")
    df_curated = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df_curated)} rows from curated dataset.")

    # Create lookup map: (fen, move_uci) -> row dict
    records_lookup = {}
    for r in df_curated.to_dict("records"):
        key = (r["fen"], r["move_uci"])
        if key not in records_lookup:
            records_lookup[key] = []
        records_lookup[key].append(r)

    splits = ["train", "val", "test"]
    expected_counts = {"train": 1942, "val": 265, "test": 292}

    print("\nInitializing warm StockfishEngine for continuation calculations...")
    with StockfishEngine(default_depth=10) as sf_engine:
        for split in splits:
            jsonl_path = f"data/commentary_sft_{split}.jsonl"
            print(f"\nProcessing {split} split ({jsonl_path})...")

            with open(jsonl_path, "r", encoding="utf-8") as f:
                original_lines = [json.loads(l) for l in f if l.strip()]

            orig_count = len(original_lines)
            assert orig_count == expected_counts[split], (
                f"Split {split} count mismatch: expected {expected_counts[split]}, got {orig_count}"
            )

            updated_records: List[Dict[str, Any]] = []
            continuation_injected_count = 0
            verification_injected_count = 0

            for entry in tqdm(original_lines, desc=f"Resyncing {split}"):
                user_msg = entry["messages"][1]["content"]
                assistant_msg = entry["messages"][2]["content"]

                fen, uci = parse_fen_and_uci_from_user_text(user_msg)
                matching_rows = records_lookup.get((fen, uci), [])

                if not matching_rows:
                    raise ValueError(f"Could not find matching parquet record for {fen} {uci}")

                if len(matching_rows) == 1:
                    row_data = matching_rows[0]
                else:
                    matched = None
                    for candidate in matching_rows:
                        if candidate.get("teacher_commentary", "").strip() == assistant_msg.strip():
                            matched = candidate
                            break
                    row_data = matched if matched is not None else matching_rows[0]

                best_move = row_data.get("best_move")
                played_best = row_data.get("played_best", False)
                if played_best is None:
                    played_best = (best_move == uci) if best_move else False

                tactic = row_data.get("tactic_type", "none")
                mistake_cat = row_data.get("mistake_category", "none")

                new_user_text = construct_user_prompt(
                    fen=fen,
                    move_san=row_data.get("move_san", ""),
                    move_uci=uci,
                    player_color=row_data.get("player_color", "white"),
                    mistake_type=row_data.get("mistake_type", "good"),
                    cp_loss=row_data.get("cp_loss"),
                    best_move=best_move,
                    played_best=bool(played_best),
                    tactic_type=tactic,
                    mistake_category=mistake_cat,
                    tactical_elements=row_data.get("tactical_elements"),
                    stockfish_engine=sf_engine,
                )

                if "Tactical Continuation:" in new_user_text:
                    continuation_injected_count += 1
                if "Tactical Elements:" in new_user_text:
                    verification_injected_count += 1

                full_text = (
                    f"<|system|>\n{SYSTEM_INSTRUCTION}<|end|>\n"
                    f"<|user|>\n{new_user_text}<|end|>\n"
                    f"<|assistant|>\n{assistant_msg}<|end|>\n"
                )

                updated_records.append({
                    "text": full_text,
                    "messages": [
                        {"role": "system", "content": SYSTEM_INSTRUCTION},
                        {"role": "user", "content": new_user_text},
                        {"role": "assistant", "content": assistant_msg},
                    ]
                })

            assert len(updated_records) == expected_counts[split]
            print(f"  {split}: {continuation_injected_count} continuation, {verification_injected_count} verification / {len(updated_records)} total.")

            temp_path = jsonl_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f_out:
                for rec in updated_records:
                    f_out.write(json.dumps(rec) + "\n")
            os.replace(temp_path, jsonl_path)
            print(f"  Successfully wrote updated {jsonl_path} ({len(updated_records)} records).")

    print("\n" + "=" * 60)
    print("VERIFICATION CHECKS")
    print("=" * 60)
    total_records = 0
    banned_phrases = [
        "CRITICAL INSTRUCTION",
        "critical instruction",
        "the instruction indicates",
        "the instruction points",
        "as instructed",
        "as provided above",
        "as given above",
        "the verified tactical continuation",
    ]
    for split in splits:
        p = f"data/commentary_sft_{split}.jsonl"
        with open(p, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f]
        count = len(lines)
        total_records += count
        assert count == expected_counts[split], f"{split} count {count} != {expected_counts[split]}"

        for idx, rec in enumerate(lines):
            assistant_content = rec["messages"][2]["content"]
            assert len(assistant_content.strip()) > 0
            for phrase in banned_phrases:
                assert phrase.lower() not in assistant_content.lower(), (
                    f"Found leakage '{phrase}' in {split} assistant message {idx}: {assistant_content}"
                )
        print(f"✓ {split}: {count} records confirmed clean of all banned scaffolding leakage phrases.")

    print(f"\nTotal records across all splits: {total_records} (expected: 2499)")
    assert total_records == 2499
    print("✓ All checks passed successfully!")


if __name__ == "__main__":
    t0 = time.time()
    resync_splits()
    print(f"Total time: {time.time() - t0:.2f}s")
