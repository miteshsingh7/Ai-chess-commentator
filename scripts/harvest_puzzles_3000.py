"""Harvest tactical puzzles from Lichess dataset via HuggingFace for the 3,000 run."""

import os
import time
import json
import urllib.request
import chess
from collections import defaultdict
from chess_commentator.analysis.taxonomy import classify_position_taxonomy

OUTPUT_PATH = "data/lichess_tactical_puzzles_3000.json"

TARGET_QUOTAS = {
    "fork": 220,
    "pin": 176,
    "skewer": 165,
    "discovered_attack": 165,
    "back_rank_mate": 110,
    "checkmate": 220,
    "trapped_piece": 80,
}

def save_current(collected):
    all_puzzles = []
    for cat_list in collected.values():
        all_puzzles.extend(cat_list)
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_puzzles, f, indent=2)

def main():
    collected = defaultdict(list)
    # Load from existing 96 puzzles first
    if os.path.exists("data/lichess_tactical_puzzles.json"):
        try:
            with open("data/lichess_tactical_puzzles.json", "r", encoding="utf-8") as f:
                existing = json.load(f)
                for item in existing:
                    cat = item.get("category")
                    if cat in TARGET_QUOTAS:
                        collected[cat].append(item)
                print(f"Loaded {len(existing)} existing baseline puzzles.")
        except Exception as e:
            print(f"Error loading existing: {e}")

    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
                for item in existing:
                    cat = item.get("category")
                    if cat in TARGET_QUOTAS and not any(p["fen"] == item["fen"] for p in collected[cat]):
                        collected[cat].append(item)
                print(f"Loaded {len(existing)} from {OUTPUT_PATH}.")
        except Exception as e:
            print(f"Error loading {OUTPUT_PATH}: {e}")

    offset = 0
    page_size = 100
    max_pages = 80

    print("Target quotas for 3,000 run:")
    for k, q in TARGET_QUOTAS.items():
        print(f"  {k:18}: {len(collected[k])}/{q}")

    while any(len(collected[k]) < TARGET_QUOTAS[k] for k in TARGET_QUOTAS) and (offset // page_size) < max_pages:
        url = f"https://datasets-server.huggingface.co/rows?dataset=Lichess/chess-puzzles&config=default&split=train&offset={offset}&limit={page_size}"
        req = urllib.request.Request(url, headers={"User-Agent": f"ChessCommentatorHarvest3000/1.0 (offset={offset})"})

        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"Fetch failed at offset {offset}: {e}. Backing off 5s...")
            time.sleep(5.0)
            offset += page_size
            continue

        rows = [item["row"] for item in data.get("rows", [])]
        if not rows:
            print("No more rows returned.")
            break

        new_added = 0
        for r in rows:
            moves = r["Moves"].split()
            if len(moves) < 2:
                continue
            themes = set(r.get("Themes", []))

            b = chess.Board(r["FEN"])
            b.push_uci(moves[0])
            fen_after_setup = b.fen()
            solver_move = chess.Move.from_uci(moves[1])

            tax = classify_position_taxonomy(
                board=b,
                move=solver_move,
                player_color=b.turn,
                eval_before=600,
                eval_after=600,
                cp_loss=0,
                best_move=solver_move,
                played_best=True,
            )

            cat = None
            t = tax.tactic_type
            if t in ("fork", "knight_fork", "pawn_fork") and ("fork" in themes):
                cat = "fork"
            elif t == "pin" and ("pin" in themes):
                cat = "pin"
            elif t == "skewer" and ("skewer" in themes):
                cat = "skewer"
            elif t == "discovered_attack" and ("discoveredAttack" in themes):
                cat = "discovered_attack"
            elif t == "back_rank_mate" and ("backRankMate" in themes or "mate" in themes):
                cat = "back_rank_mate"
            elif t == "checkmate" and ("mateIn1" in themes or "mate" in themes):
                cat = "checkmate"
            elif (t in ("trapped_piece", "trapped_bishop", "trapped_knight", "trapped_rook") or "trappedPiece" in themes):
                cat = "trapped_piece"

            if cat and len(collected[cat]) < TARGET_QUOTAS[cat]:
                if not any(p["fen"] == fen_after_setup for p in collected[cat]):
                    collected[cat].append({
                        "puzzle_id": r["PuzzleId"],
                        "fen": fen_after_setup,
                        "move_uci": solver_move.uci(),
                        "move_san": b.san(solver_move),
                        "rating": r.get("Rating"),
                        "themes": list(themes),
                        "tax_tactic_type": t,
                        "category": cat,
                    })
                    new_added += 1

        if new_added > 0:
            save_current(collected)

        offset += page_size
        counts_str = ", ".join(f"{k}: {len(collected[k])}/{TARGET_QUOTAS[k]}" for k in TARGET_QUOTAS)
        print(f"Offset {offset:4d} | {counts_str}")
        time.sleep(1.0)

    save_current(collected)
    total_saved = sum(len(v) for v in collected.values())
    print(f"\nSaved {total_saved} tactical puzzles to {OUTPUT_PATH}:")
    for k, q in TARGET_QUOTAS.items():
        print(f"  {k:18}: {len(collected[k])}/{q}")

if __name__ == "__main__":
    main()
