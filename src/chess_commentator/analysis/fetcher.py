"""Chess.com API client for fetching monthly game archives in PGN format.

Adapted from chessIQ phase1_fetch_games.py.
"""

import json
import os
import time
from typing import List, Dict, Any, Optional
import requests


def fetch_player_games(
    username: str,
    output_dir: str = "data/raw_pgn",
    max_months: Optional[int] = None,
    time_class: str = "rapid",
) -> List[Dict[str, Any]]:
    """Fetch monthly PGN archives for a given Chess.com username."""
    os.makedirs(output_dir, exist_ok=True)
    headers = {"User-Agent": "AIChessCommentator/1.0 (https://github.com/miteshsingh7/chess-commentator)"}

    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    response = requests.get(archives_url, headers=headers, timeout=15)
    response.raise_for_status()
    archives = response.json().get("archives", [])

    if max_months is not None:
        archives = archives[-max_months:]

    all_games: List[Dict[str, Any]] = []

    for archive_url in archives:
        month_label = archive_url.split("/")[-2] + "_" + archive_url.split("/")[-1]
        pgn_path = os.path.join(output_dir, f"{username}_{month_label}.pgn")
        meta_path = os.path.join(output_dir, f"{username}_{month_label}_meta.json")

        if os.path.exists(pgn_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    all_games.extend(json.load(f))
                continue
            except (json.JSONDecodeError, IOError):
                os.remove(pgn_path)

        time.sleep(0.4)
        try:
            games_resp = requests.get(archive_url, headers=headers, timeout=20)
            games_resp.raise_for_status()
            games = games_resp.json().get("games", [])
        except requests.RequestException:
            continue

        selected_games = [g for g in games if g.get("time_class") == time_class]

        with open(pgn_path, "w", encoding="utf-8") as f:
            for g in selected_games:
                if "pgn" in g:
                    f.write(g["pgn"] + "\n\n")

        meta = []
        for g in selected_games:
            meta.append({
                "url": g.get("url"),
                "time_control": g.get("time_control"),
                "end_time": g.get("end_time"),
                "rated": g.get("rated"),
                "time_class": g.get("time_class"),
                "rules": g.get("rules"),
                "white": g.get("white", {}),
                "black": g.get("black", {}),
            })

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        all_games.extend(meta)

    return all_games
