"""Unit tests for Stage 3: Dataset filtering, curation, balancing, and formatting."""

import tempfile
import pandas as pd
import pytest

from chess_commentator.dataset.filters import (
    validate_eval_sign_consistency,
    validate_filler_and_length,
    PhraseDiversityChecker,
)
from chess_commentator.dataset.checker import validate_chess_grounding
from chess_commentator.dataset.balancer import (
    deduplicate_dataset,
    balance_taxonomy_classes,
    balance_and_split_dataset,
)
from chess_commentator.dataset.formatter import row_to_chatml, export_sft_dataset


def test_eval_sign_filter():
    # Blunder praised -> must fail
    res1 = validate_eval_sign_consistency(
        commentary="This is a brilliant move that wins cleanly.",
        cp_loss=400,
        played_best=False,
        mistake_type="blunder",
    )
    assert res1.passed is False
    assert "contradiction" in res1.reason.lower()

    # Good move called a blunder -> must fail
    res2 = validate_eval_sign_consistency(
        commentary="This terrible blunder ruins the entire game.",
        cp_loss=0,
        played_best=True,
        mistake_type="good",
    )
    assert res2.passed is False

    # Valid commentary -> passes
    res3 = validate_eval_sign_consistency(
        commentary="A costly mistake that drops a rook to the knight fork.",
        cp_loss=350,
        played_best=False,
        mistake_type="blunder",
    )
    assert res3.passed is True


def test_filler_and_length_filter():
    # Robotic AI filler
    res1 = validate_filler_and_length(
        "As an AI chess commentator, this move is played on the board in this position."
    )
    assert res1.passed is False
    assert "filler" in res1.reason.lower()

    # Too short
    res2 = validate_filler_and_length("Bad move.")
    assert res2.passed is False
    assert "short" in res2.reason.lower()

    # Clean commentary
    clean_text = (
        "White plays a sharp tactical blow, forcing Black's king out into the open. "
        "The queen and bishop coordinate seamlessly to prevent any escape squares."
    )
    res3 = validate_filler_and_length(clean_text)
    assert res3.passed is True


def test_chess_grounding_checker():
    # Starting position: White knight is on g1 or b1. Claiming "knight on f7" is a hallucination.
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    res_bad = validate_chess_grounding(
        fen=fen,
        move_uci="e2e4",
        commentary="White uses the knight on f7 to deliver devastating checks.",
    )
    assert res_bad.passed is False
    assert "hallucination" in res_bad.reason.lower()

    # Valid piece mention: knight on g1
    res_good = validate_chess_grounding(
        fen=fen,
        move_uci="g1f3",
        commentary="White develops the knight to f3, controlling central squares and preparing kingside castling.",
    )
    assert res_good.passed is True


def test_chess_grounding_negative_assertion():
    # In this position, f7 has NO pawn (it's empty).
    fen_mate = "r2qrk2/p2n2p1/b1pb1n1p/8/1p1P1N2/1Q3N1P/PP3PP1/R1B1R1K1 w - - 2 17"
    # Grounded: correctly states absence of a pawn on f7
    comm_valid = "White delivers Ng6#, exploiting the lack of a pawn on f7 to block the check."
    res_valid = validate_chess_grounding(
        fen=fen_mate,
        move_uci="f4g6",
        commentary=comm_valid,
    )
    assert res_valid.passed is True

    # Hallucinated negative assertion: claiming absence of pawn on a7 (which IS present!)
    comm_invalid = "White attacks due to the absence of a pawn on a7."
    res_invalid = validate_chess_grounding(
        fen=fen_mate,
        move_uci="f4g6",
        commentary=comm_invalid,
    )
    assert res_invalid.passed is False
    assert "hallucination" in res_invalid.reason.lower()


def test_chess_grounding_checker_catches_illegal_commentary_moves():
    """Assert validate_chess_grounding catches illegal/hallucinated move notations in commentary.
    
    Specifically tests the pgn_259 regression where the teacher claimed Black responds with ...Bxf1+,
    which is an illegal move for Black.
    """
    fen_259 = "rnbqkbnr/pp3ppp/8/4N3/3p4/2N5/PP1PPPPP/R1BQKB1R w KQkq - 0 6"
    played_259 = "c3b1"
    bm_259 = "d1a4"

    # 1. Illegal move (...Bxf1+) -> must fail
    comm_illegal = (
        "Retreating the knight to b1 is a severe blunder that abandons the ideal c3 outpost "
        "and allows Black to seize the initiative with the zwischenzug ...Bxf1+. "
        "White overlooked this forcing sequence."
    )
    res_fail = validate_chess_grounding(
        fen=fen_259,
        move_uci=played_259,
        commentary=comm_illegal,
        best_move_uci=bm_259,
    )
    assert res_fail.passed is False
    assert "illegal move" in res_fail.reason.lower()
    assert "bxf1+" in res_fail.reason.lower()

    # 2. Fully legal moves (Qa4, d1a4, c3b1, e7e6, etc.) -> must pass
    comm_legal = (
        "Retreating the knight to b1 is a passive move that forfeits central control. "
        "The engine strongly recommends Qa4, immediately pressuring the d4 pawn and pinning Black's pieces."
    )
    res_pass = validate_chess_grounding(
        fen=fen_259,
        move_uci=played_259,
        commentary=comm_legal,
        best_move_uci=bm_259,
    )
    assert res_pass.passed is True

    data = []
    for i in range(100):
        data.append({
            "game_id": f"game_{i // 10}",
            "fen": f"fen_{i}",
            "move_uci": f"move_{i}",
            "tactic_type": "fork" if i % 2 == 0 else "pin",
            "teacher_commentary": f"Commentary text for position {i} explaining tactical nuances clearly.",
            "mistake_type": "blunder" if i % 3 == 0 else "good",
        })
    df = pd.DataFrame(data)

    # Test deduplication
    df_dup = pd.concat([df, df.iloc[:10]], ignore_index=True)
    assert len(df_dup) == 110
    assert len(deduplicate_dataset(df_dup)) == 100

    # Test allow_multisample with distinct commentaries
    df_resample = df.iloc[:5].copy()
    df_resample["teacher_commentary"] = df_resample["teacher_commentary"] + " (alternative wording)"
    df_combined = pd.concat([df, df_resample], ignore_index=True)
    assert len(df_combined) == 105
    # Default (allow_multisample=False) drops all FEN/move duplicates
    assert len(deduplicate_dataset(df_combined, allow_multisample=False)) == 100
    # allow_multisample=True preserves genuinely distinct temperature generations
    assert len(deduplicate_dataset(df_combined, allow_multisample=True)) == 105

    # Test bounded oversampling in balance_taxonomy_classes
    df_tiny = pd.DataFrame([
        {"tactic_type": "rare", "fen": f"f{i}", "move_uci": f"m{i}"} for i in range(5)
    ])
    # Target 50 with max_oversample_factor=2.0 must cap at 5 * 2 = 10
    df_bounded = balance_taxonomy_classes(
        df_tiny, min_per_category=50, max_oversample_factor=2.0, random_seed=42
    )
    assert len(df_bounded) == 10

    # Test split
    train, val, test = balance_and_split_dataset(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    # Ensure no game overlap between train and test
    train_games = set(train["game_id"].unique())
    test_games = set(test["game_id"].unique())
    assert len(train_games.intersection(test_games)) == 0


def test_chatml_formatting_and_export():
    row = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "move_san": "e4",
        "move_uci": "e2e4",
        "player_color": "white",
        "cp_loss": 0,
        "mistake_type": "good",
        "tactic_type": "none",
        "teacher_commentary": "Strong central thrust claiming key squares.",
    }
    chatml = row_to_chatml(row)
    assert len(chatml["messages"]) == 3
    assert chatml["messages"][0]["role"] == "system"
    assert chatml["messages"][1]["role"] == "user"
    assert chatml["messages"][2]["role"] == "assistant"
    assert chatml["messages"][2]["content"] == "Strong central thrust claiming key squares."

    # Test JSONL file export
    import os
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        export_sft_dataset(pd.DataFrame([row]), tf.name)
        assert os.path.getsize(tf.name) > 0


def test_phrase_diversity_checker():
    checker = PhraseDiversityChecker(n=3, threshold=0.40)

    # 1. First entry in bucket -> never flagged
    text1 = (
        "The Caro-Kann Defense is a solid classical choice that prepares to challenge "
        "White's central pawn on e4 with d5 while maintaining a robust pawn structure."
    )
    res1 = checker.check_and_add(text1, bucket="none")
    assert res1.is_flagged is False
    assert res1.max_overlap == 0.0

    # 2. Diverse second entry in same bucket -> passes
    text2 = (
        "Black strikes dynamically at the center with a forceful pawn push, contesting "
        "key outposts and opening active development diagonals for the bishop."
    )
    res2 = checker.check_and_add(text2, bucket="none")
    assert res2.is_flagged is False
    assert res2.max_overlap < 0.40

    # 3. Near-duplicate templated entry with only one word changed -> must be flagged (> 40% overlap)
    text3 = (
        "The French Defense is a solid classical choice that prepares to challenge "
        "White's central pawn on e4 with d5 while maintaining a robust pawn structure."
    )
    res3 = checker.check_and_add(text3, bucket="none")
    assert res3.is_flagged is True
    assert res3.max_overlap > 0.40
    assert res3.matched_reference == text1

    # 4. Same text in a DIFFERENT bucket -> not flagged against other buckets
    res4 = checker.check_and_add(text3, bucket="fork")
    assert res4.is_flagged is False

    # 5. Test dataframe audit
    df = pd.DataFrame([
        {"teacher_commentary": text1, "tactic_type": "none"},
        {"teacher_commentary": text2, "tactic_type": "none"},
        {"teacher_commentary": text3, "tactic_type": "none"},
    ])
    audit = checker.audit_dataframe(df)
    assert audit["total_checked"] == 3
    assert audit["flagged_count"] == 1
    assert len(audit["flagged_records"]) == 1
    assert audit["flagged_records"][0]["index"] == 2


def test_chess_grounding_unicode_hyphen_normalization():
    """Verify that non-ASCII hyphens (\u2011, \u2013, \u2212) in move references normalize to ASCII - and pass."""
    fen = "r3r1k1/1b2Pq1p/p1p3p1/1p6/P3PPBB/2PQ3P/6P1/4R1K1 b - - 0 23"
    played_uci = "f7f4"
    best_uci = "e8e7"

    # 1. Non-breaking hyphen (\u2011)
    comm_nb_hyphen = (
        "Qxf4 wins a pawn but abandons the decisive ...e8\u2011e7 rook sacrifice. "
        "The engine recommends ...e8\u2011e7 to dismantle White's position."
    )
    res1 = validate_chess_grounding(fen, played_uci, comm_nb_hyphen, best_uci)
    assert res1.passed is True, f"Failed on non-breaking hyphen: {res1.reason}"

    # 2. En-dash (\u2013) and minus sign (\u2212)
    comm_dash = "White could have continued with e8\u2013e7 or e8\u2212e7, which is completely winning."
    res2 = validate_chess_grounding(fen, played_uci, comm_dash, best_uci)
    assert res2.passed is True, f"Failed on dash/minus: {res2.reason}"


def test_chess_grounding_ply2_king_destination_after_check():
    """Verify that ply+2 destinations in the position tree (e.g. king forced to square after check) pass grounding."""
    fen = "6k1/2Q5/2p4p/5p1q/P1P1p3/1P1rB2P/6K1/5R2 w - - 4 43"
    played_uci = "c7f4"  # Qf4
    best_uci = "c7g3"    # Qg3+

    # King to h7 is a legal ply-2 response for Black after White's Qg3+ check
    comm_legal_dest = (
        "Qf4 is a critical mistake because it misses the forcing zwischenzug Qg3+, "
        "which forces the black king to h7 and keeps White firmly in control."
    )
    res_pass = validate_chess_grounding(fen, played_uci, comm_legal_dest, best_uci)
    assert res_pass.passed is True, f"Expected ply+2 king destination to pass, got: {res_pass.reason}"

    # King to a1 is NOT a legal destination for Black's king anywhere in this tree -> must fail
    comm_illegal_dest = (
        "White should play Qg3+, which forces the black king to a1 and wins on the spot."
    )
    res_fail = validate_chess_grounding(fen, played_uci, comm_illegal_dest, best_uci)
    assert res_fail.passed is False
    assert "hallucination detected" in res_fail.reason.lower()
    assert "a1" in res_fail.reason.lower()


def test_stratified_game_aware_splitting():
    """Verify that thin categories are guaranteed representation across train, val, and test splits without game leakage."""
    records = []
    # 20 games of common category "fork"
    for i in range(20):
        records.append({"game_id": f"game_fork_{i}", "fen": f"fen_f_{i}", "move_uci": "e2e4", "tactic_type": "fork"})
    # 3 games of thin category "zwischenzug"
    for i in range(3):
        records.append({"game_id": f"game_zw_{i}", "fen": f"fen_z_{i}", "move_uci": "d2d4", "tactic_type": "zwischenzug"})
    # 2 games of ultra-thin category "trapped_queen"
    for i in range(2):
        records.append({"game_id": f"game_tq_{i}", "fen": f"fen_q_{i}", "move_uci": "c2c4", "tactic_type": "trapped_queen"})

    df = pd.DataFrame(records)
    train_df, val_df, test_df = balance_and_split_dataset(
        df,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        min_per_category=3,
        random_seed=42,
    )

    # 1. Zero game leakage
    tr_g = set(train_df["game_id"])
    va_g = set(val_df["game_id"])
    te_g = set(test_df["game_id"])
    assert len(tr_g & va_g) == 0
    assert len(tr_g & te_g) == 0
    assert len(va_g & te_g) == 0

    # 2. Representation of thin categories across all three splits
    for cat in ["fork", "zwischenzug", "trapped_queen"]:
        assert (train_df["tactic_type"] == cat).sum() >= 1, f"Missing {cat} in train"
        assert (test_df["tactic_type"] == cat).sum() >= 1, f"Missing {cat} in test"
        assert (val_df["tactic_type"] == cat).sum() >= 1, f"Missing {cat} in val"


def test_validate_no_scaffolding_leakage():
    from chess_commentator.dataset.filters import validate_no_scaffolding_leakage

    # Banned scaffolding phrases must be rejected
    banned_samples = [
        "The critical instruction points to a tactical motif where Black's pawn on g7 is hanging.",
        "The instruction indicates that without further moves, Black's move is flawed.",
        "As instructed, White plays the engine top move.",
        "The verified tactical continuation shows that Black is completely winning.",
        "According to the continuation provided above, White has a winning sacrifice.",
    ]
    for sample in banned_samples:
        res = validate_no_scaffolding_leakage(sample)
        assert res.passed is False, f"Failed to reject scaffolding: {sample}"
        assert "scaffolding leakage" in res.reason.lower()

    # Natural chess text containing words like 'critical' or 'continuation' must PASS
    valid_samples = [
        "White retains the initiative in the critical endgame phase by driving the king away.",
        "The move is a precise continuation of the attack, ensuring Black cannot consolidate.",
        "Controlling the critical d5 square gives White a decisive positional advantage.",
    ]
    for sample in valid_samples:
        res = validate_no_scaffolding_leakage(sample)
        assert res.passed is True, f"Incorrectly rejected valid chess commentary: {sample}"



