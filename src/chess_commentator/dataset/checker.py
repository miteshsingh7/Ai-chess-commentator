"""Chess grounding and hallucination validation for commentary text."""

import re
from typing import Optional, List, Tuple
import chess
from chess_commentator.dataset.filters import FilterResult


SQUARE_REGEX = re.compile(r'\b([a-h][1-8])\b', re.IGNORECASE)
PIECE_PATTERN = re.compile(
    r'\b(queen|rook|bishop|knight|pawn|king)\s+(?:on|at|to|from)\s+([a-h][1-8])\b',
    re.IGNORECASE,
)

PIECE_MAP = {
    "queen": chess.QUEEN,
    "rook": chess.ROOK,
    "bishop": chess.BISHOP,
    "knight": chess.KNIGHT,
    "pawn": chess.PAWN,
    "king": chess.KING,
}

UNICODE_HYPHENS = re.compile(r'[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]')

# Regex for SAN and algebraic moves
HYPHEN_MOVE_PATTERN = re.compile(
    r'(?<!\w)'
    r'(?:(?P<prefix>\d+\.+|\.{2,3})\s*)?'
    r'(?P<san>[NBRQK]?[a-h][1-8][-–][a-h][1-8][+#]?)'
    r'(?!\w)',
    re.UNICODE,
)

STANDARD_SAN_PATTERN = re.compile(
    r'(?<!\w)'
    r'(?:(?P<prefix>\d+\.+|\.{2,3})\s*)?'
    r'(?P<san>'
    r'O-O(?:-O)?[+#]?'
    r'|[NBRQK][a-h1-8]?x?[a-h][1-8](?:=[NBRQK])?[+#]?'
    r'|[a-h]x[a-h][1-8](?:=[NBRQK])?[+#]?'
    r'|[a-h][1-8]=[NBRQK][+#]?'
    r'|[a-h][1-8][+#]'
    r')(?!\w)',
    re.UNICODE,
)

EXPLICIT_PAWN_PATTERN = re.compile(
    r'(?:\b(?:move|plays|played)\s+|(?:\d+\.+|\.{2,3})\s*)([a-h][1-8])\b',
    re.IGNORECASE,
)

NEGATIVE_ASSERTION_PREFIX = re.compile(
    r'\b(?:absence\s+of|lack\s+of|without\s+a|without\s+any|without|no|neither)\s+(?:a\s+|an\s+|the\s+|any\s+)?$',
    re.IGNORECASE,
)


def extract_piece_square_mentions(commentary: str) -> List[Tuple[str, str, bool]]:
    """Extract explicit claims like 'knight on f3' or negative claims like 'absence of a pawn on f7'.

    Returns list of (piece_name, sq_name, is_negative_assertion).
    """
    mentions = []
    for m in PIECE_PATTERN.finditer(commentary):
        piece_name = m.group(1).lower()
        sq_name = m.group(2).lower()
        preceding = commentary[max(0, m.start() - 35):m.start()]
        is_neg = bool(NEGATIVE_ASSERTION_PREFIX.search(preceding))
        mentions.append((piece_name, sq_name, is_neg))
    return mentions


def extract_algebraic_move_references(commentary: str) -> List[Tuple[str, str]]:
    """Extract candidate chess move notations (SAN, UCI, long algebraic) from commentary.

    Returns list of (clean_san_or_uci, raw_token).
    """
    # Normalize unicode hyphens early so downstream patterns recognize non-breaking hyphens, en/em dashes, etc.
    commentary = UNICODE_HYPHENS.sub('-', commentary)

    moves = []

    # 1. Hyphenated long algebraic (e.g. Kg1-h2, c7-h2, Rf1-c1+, ...e8-e7)
    for m in HYPHEN_MOVE_PATTERN.finditer(commentary):
        san = m.group('san')
        raw = m.group(0)

        # Skip if part of a multi-hyphen sequence (e.g. b7-c6-d5-e4)
        if (m.start() > 0 and commentary[m.start() - 1] == '-') or (m.end() < len(commentary) and commentary[m.end()] == '-'):
            continue

        # Skip if describing a line/diagonal/range/tension e.g. "a1-h8 diagonal", "controls d2-d3", "d5-e4 tension"
        following = commentary[m.end():m.end() + 20].lower()
        preceding = commentary[max(0, m.start() - 20):m.start()].lower()
        if (
            any(w in following for w in ["diag", "file", "rank", "line", "tension", "square"]) or
            any(w in preceding for w in ["along", "diagonal", "control", "cover", "guard", "defend", "tension", "between", "square"])
        ):
            continue
        moves.append((san, raw, m.start()))

    # 2. Standard SAN moves (e.g. Bxf1+, Qa4+, Nxe7+, Qxf7#, O-O, cxd4)
    for m in STANDARD_SAN_PATTERN.finditer(commentary):
        san = m.group('san')
        raw = m.group(0)
        moves.append((san, raw, m.start()))

    # 3. Explicit pawn moves (e.g. "move f4", "plays e4", "1. e4", "...f5")
    for m in EXPLICIT_PAWN_PATTERN.finditer(commentary):
        san = m.group(1)
        raw = m.group(0)
        moves.append((san, raw, m.start()))

    # Deduplicate overlapping matches by character span
    unique_moves = []
    seen_spans: List[Tuple[int, int]] = []
    for san, raw, start in sorted(moves, key=lambda x: x[2]):
        end = start + len(raw)
        if any(not (end <= s or start >= e) for s, e in seen_spans):
            continue
        seen_spans.append((start, end))
        clean_san = san.strip().lstrip('.').rstrip('!?')
        unique_moves.append((clean_san, raw.strip()))

    return unique_moves


def is_legal_notation_on_board(b: chess.Board, notation: str) -> bool:
    """Check if notation is legal on board b."""
    hyphen_m = re.match(r'^([NBRQK]?)([a-h][1-8])[-–]([a-h][1-8])([+#]?)$', notation)
    if hyphen_m:
        from_sq = chess.parse_square(hyphen_m.group(2))
        to_sq = chess.parse_square(hyphen_m.group(3))
        candidate_move = chess.Move(from_sq, to_sq)
        if candidate_move in b.legal_moves:
            return True
        for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            if chess.Move(from_sq, to_sq, promotion=promo) in b.legal_moves:
                return True
        return False

    try:
        m = b.parse_san(notation)
        if m in b.legal_moves:
            return True
    except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError):
        pass

    try:
        m = chess.Move.from_uci(notation)
        if m in b.legal_moves:
            return True
    except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError):
        pass

    # Destination square shorthand (e.g. "...e7" or "move e7" where piece prefix was omitted in prose)
    if re.match(r'^[a-h][1-8]$', notation):
        sq = chess.parse_square(notation)
        if any(m.to_square == sq for m in b.legal_moves):
            return True

    return False


def is_destination_for_piece(b: chess.Board, expected_type: chess.PieceType, target_sq: chess.Square) -> bool:
    """Check if target_sq is a legal destination for expected_type on board b."""
    for m in b.legal_moves:
        if m.to_square == target_sq:
            p = b.piece_at(m.from_square)
            if p and (p.piece_type == expected_type or m.promotion == expected_type):
                return True
        if b.is_castling(m) and expected_type == chess.ROOK:
            castling_rook_dest = {
                chess.G1: chess.F1,
                chess.C1: chess.D1,
                chess.G8: chess.F8,
                chess.C8: chess.D8,
            }.get(m.to_square)
            if castling_rook_dest == target_sq:
                return True
    return False


def is_move_legal_in_position_tree(
    board_before: chess.Board,
    played_move: Optional[chess.Move],
    best_move: Optional[chess.Move],
    notation: str,
) -> bool:
    """Verify whether a move notation is legal in the relevant game tree (ply 0 to 3)."""
    # 1. Ply 1: Legal for active player on board_before (played move, best move, alternative)
    if is_legal_notation_on_board(board_before, notation):
        return True

    # 1b. Ply 0: Opponent's previous threat
    b_opp = board_before.copy()
    b_opp.turn = not board_before.turn
    if is_legal_notation_on_board(b_opp, notation):
        return True

    # 2. Ply 2: Opponent's response after played move
    if played_move and played_move in board_before.legal_moves:
        board_after = board_before.copy()
        board_after.push(played_move)
        if is_legal_notation_on_board(board_after, notation):
            return True

        # 3. Ply 3: Follow-up after played move + opponent reply
        for opp_m in list(board_after.legal_moves)[:30]:
            b_ply3 = board_after.copy()
            b_ply3.push(opp_m)
            if is_legal_notation_on_board(b_ply3, notation):
                return True

    # 4. Ply 2: Opponent's response after best move
    if best_move and best_move in board_before.legal_moves:
        board_best = board_before.copy()
        board_best.push(best_move)
        if is_legal_notation_on_board(board_best, notation):
            return True

        # 5. Ply 3: Follow-up after best move + opponent reply
        for opp_m in list(board_best.legal_moves)[:30]:
            b_best_ply3 = board_best.copy()
            b_best_ply3.push(opp_m)
            if is_legal_notation_on_board(b_best_ply3, notation):
                return True

    return False


def is_piece_square_grounded_in_tree(
    board_before: chess.Board,
    played_move: Optional[chess.Move],
    best_move: Optional[chess.Move],
    expected_type: chess.PieceType,
    target_sq: chess.Square,
) -> bool:
    """Check if piece is present or legally moves to target_sq across ply 0 to ply 3 tree."""
    # 1. Present on board_before
    p = board_before.piece_at(target_sq)
    if p and p.piece_type == expected_type:
        return True

    # 2. Legal move on board_before (Ply 1) or b_opp (Ply 0)
    if is_destination_for_piece(board_before, expected_type, target_sq):
        return True
    b_opp = board_before.copy()
    b_opp.turn = not board_before.turn
    if is_destination_for_piece(b_opp, expected_type, target_sq):
        return True

    # 3. Played move tree (Ply 2 and Ply 3)
    if played_move and played_move in board_before.legal_moves:
        board_after = board_before.copy()
        board_after.push(played_move)
        p = board_after.piece_at(target_sq)
        if p and p.piece_type == expected_type:
            return True
        if is_destination_for_piece(board_after, expected_type, target_sq):
            return True
        for opp_m in list(board_after.legal_moves)[:30]:
            b_ply3 = board_after.copy()
            b_ply3.push(opp_m)
            p3 = b_ply3.piece_at(target_sq)
            if p3 and p3.piece_type == expected_type:
                return True
            if is_destination_for_piece(b_ply3, expected_type, target_sq):
                return True

    # 4. Best move tree (Ply 2 and Ply 3)
    if best_move and best_move in board_before.legal_moves:
        board_best = board_before.copy()
        board_best.push(best_move)
        p = board_best.piece_at(target_sq)
        if p and p.piece_type == expected_type:
            return True
        if is_destination_for_piece(board_best, expected_type, target_sq):
            return True
        for opp_m in list(board_best.legal_moves)[:30]:
            b_best_ply3 = board_best.copy()
            b_best_ply3.push(opp_m)
            p3 = b_best_ply3.piece_at(target_sq)
            if p3 and p3.piece_type == expected_type:
                return True
            if is_destination_for_piece(b_best_ply3, expected_type, target_sq):
                return True

    return False


def validate_chess_grounding(
    fen: str,
    move_uci: str,
    commentary: str,
    best_move_uci: Optional[str] = None,
) -> FilterResult:
    """Verify that pieces and moves referenced in commentary actually exist and are legal."""
    try:
        board_before = chess.Board(fen)
        move = chess.Move.from_uci(move_uci) if move_uci else None
        best_move = chess.Move.from_uci(best_move_uci) if best_move_uci else None
    except Exception as e:
        return FilterResult(passed=False, reason=f"Invalid chess state: {e}")

    # 1. Piece-square mentions validation across full position tree (ply 0 through 3)
    mentions = extract_piece_square_mentions(commentary)
    for piece_name, sq_name, is_negative in mentions:
        target_sq = chess.parse_square(sq_name)
        expected_type = PIECE_MAP.get(piece_name)
        if expected_type is None:
            continue

        if is_negative:
            # Negative assertion (e.g. "absence of a pawn on f7"):
            # Grounded if the square does NOT currently hold that piece.
            p_current = board_before.piece_at(target_sq)
            if p_current and p_current.piece_type == expected_type:
                return FilterResult(
                    passed=False,
                    reason=f"Hallucination detected: Claimed absence of {piece_name} on {sq_name}, but {piece_name} is present",
                )
        else:
            if not is_piece_square_grounded_in_tree(board_before, move, best_move, expected_type, target_sq):
                return FilterResult(
                    passed=False,
                    reason=f"Hallucination detected: No {piece_name} on or moving to square {sq_name}",
                )

    # 2. Algebraic move references validation
    move_refs = extract_algebraic_move_references(commentary)
    for clean_san, raw_token in move_refs:
        if not is_move_legal_in_position_tree(board_before, move, best_move, clean_san):
            return FilterResult(
                passed=False,
                reason=f"Illegal move mentioned in commentary: '{raw_token}' is not legal in this position tree",
            )

    return FilterResult(passed=True)

