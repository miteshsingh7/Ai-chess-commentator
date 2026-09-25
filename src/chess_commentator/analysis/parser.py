"""PGN parsing and move sequence extraction.

Adapted from chessIQ phase1_parse_pgn.py. Stripped of UI dependencies.
"""

import os
import re
from io import StringIO
from typing import List, Dict, Any, Optional
import chess
import chess.pgn


def parse_clock(comment: str) -> Optional[int]:
    """Extract clock time in seconds from PGN comment like [%clk 0:05:23]."""
    match = re.search(r'\[%clk\s+(\d+):(\d+):(\d+)\]', comment)
    if match:
        h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return h * 3600 + m * 60 + s
    return None


def get_game_result(game: chess.pgn.Game, player_username: str) -> str:
    """Return 'win', 'loss', or 'draw' from the perspective of player_username."""
    result = game.headers.get("Result", "*")
    white = game.headers.get("White", "").lower()
    black = game.headers.get("Black", "").lower()
    player = player_username.lower()

    if result == "1-0":
        return "win" if white == player else "loss"
    if result == "0-1":
        return "win" if black == player else "loss"
    if result == "1/2-1/2":
        return "draw"
    return "unknown"


def parse_pgn_string(
    pgn_text: str,
    player_username: Optional[str] = None,
    source_name: str = "pgn",
) -> List[Dict[str, Any]]:
    """Parse a PGN string containing one or more games into a list of move records."""
    rows: List[Dict[str, Any]] = []
    pgn_io = StringIO(pgn_text)
    game_id = 0

    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break

        game_id += 1
        headers = game.headers
        white = headers.get("White", "")
        black = headers.get("Black", "")

        track_all = player_username is None
        player_clean = player_username.lower() if player_username else ""
        if not track_all and player_clean not in (white.lower(), black.lower()):
            continue

        result = get_game_result(game, player_clean) if player_username else headers.get("Result", "*")
        time_control = headers.get("TimeControl", "?")
        eco = headers.get("ECO", "?")
        opening = headers.get("Opening", "?")
        white_elo = headers.get("WhiteElo")
        black_elo = headers.get("BlackElo")

        board = game.board()
        move_number = 0

        for node in game.mainline():
            move = node.move
            fen_before = board.fen()
            san = board.san(move)
            color = board.turn
            color_str = "white" if color == chess.WHITE else "black"

            clock_time = parse_clock(node.comment) if node.comment else None
            move_number += 1

            should_record = (
                track_all or
                (color == chess.WHITE and white.lower() == player_clean) or
                (color == chess.BLACK and black.lower() == player_clean)
            )

            if should_record:
                rows.append({
                    "game_id": f"{source_name}_{game_id}",
                    "move_number": move_number,
                    "fen": fen_before,
                    "move_san": san,
                    "move_uci": move.uci(),
                    "time_left": clock_time,
                    "player_color": color_str,
                    "result": result,
                    "time_control": time_control,
                    "eco": eco,
                    "opening": opening,
                    "white_player": white,
                    "black_player": black,
                    "white_elo": white_elo,
                    "black_elo": black_elo,
                })

            board.push(move)

    return rows


def parse_pgn_file(
    pgn_path: str,
    player_username: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Parse a PGN file on disk and return move-level records."""
    source_name = os.path.splitext(os.path.basename(pgn_path))[0]
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return parse_pgn_string(content, player_username, source_name)
