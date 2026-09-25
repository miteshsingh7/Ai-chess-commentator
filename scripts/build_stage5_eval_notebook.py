"""Script to generate the Kaggle notebook for Stage 5 full test set evaluation."""

import base64
import gzip
import json
from pathlib import Path


def build_notebook() -> None:
    test_inputs_path = Path("data/stage5_test_inputs.json")
    with open(test_inputs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} test items from {test_inputs_path}")

    # Compress test inputs with gzip + base64
    raw_bytes = gzip.compress(json.dumps(data).encode("utf-8"))
    b64_str = base64.b64encode(raw_bytes).decode("ascii")
    print(f"Compressed data size: {len(b64_str)} chars")

    cell_0_code = """# 1. Install dependencies
!pip install -q bitsandbytes peft python-chess
"""

    cell_1_code = """# 2. Find adapter and load model
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "microsoft/Phi-3.5-mini-instruct"

adapter_dir = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "adapter_config.json" in files:
        adapter_dir = root
        break

print(f"Using adapter_dir: {adapter_dir}")
assert adapter_dir is not None, "Adapter directory not found in /kaggle/input!"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=False,
)

print(f"Attaching fine-tuned LoRA adapter from {adapter_dir}...")
model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()
model.config.use_cache = False
print("Model ready for inference!")
"""

    cell_2_code = """# 3. Grounding, Scaffolding Leakage, and NLP Metrics Implementation
import re, json, chess
from dataclasses import dataclass
from typing import Optional, List, Tuple, Set, Final

@dataclass(frozen=True)
class FilterResult:
    passed: bool
    reason: str = "ok"

# Banned scaffolding or self-referential prompt leakage patterns
BANNED_SCAFFOLDING_PATTERNS = [
    re.compile(r"\\bcritical\\s+instruction\\b", re.IGNORECASE),
    re.compile(r"\\binstruction\\s+(?:indicates|points|states|notes|requires|suggests)\\b", re.IGNORECASE),
    re.compile(r"\\bas\\s+(?:instructed|provided\\s+above|given\\s+above|shown\\s+above)\\b", re.IGNORECASE),
    re.compile(r"\\btactical\\s+continuation\\s+(?:shows|indicates|points|suggests)\\b", re.IGNORECASE),
    re.compile(r"\\b(?:the\\s+)?verified\\s+(?:tactical\\s+)?continuation\\b", re.IGNORECASE),
    re.compile(r"\\bcontinuation\\s+(?:provided|given|listed|above)\\b", re.IGNORECASE),
    re.compile(r"\\b(?:scaffolding|system\\s+prompt)\\b", re.IGNORECASE),
]

def validate_no_scaffolding_leakage(commentary: str) -> FilterResult:
    for pat in BANNED_SCAFFOLDING_PATTERNS:
        match = pat.search(commentary)
        if match:
            return FilterResult(passed=False, reason=f"Scaffolding leakage detected: '{match.group(0)}'")
    return FilterResult(passed=True)

# Positive praise words that must NEVER describe a major blunder
PRAISE_WORDS: Set[str] = {
    "brilliant", "masterpiece", "excellent", "fantastic", "superb",
    "flawless", "great move", "very strong move", "masterful", "wonderful move", "solid improvement",
}

# Blunder words that must NEVER describe a good / best move
BLUNDER_WORDS: Set[str] = {
    "blunder", "terrible", "disastrous", "horrible", "severe error",
    "throws away", "catastrophic", "blunders away",
}

def validate_eval_sign_consistency(
    commentary: str,
    cp_loss: Optional[int],
    played_best: bool,
    mistake_type: str,
) -> FilterResult:
    comm_lower = commentary.lower()
    is_blunder = (cp_loss is not None and cp_loss >= 300) or (mistake_type == "blunder")
    if is_blunder:
        for word in PRAISE_WORDS:
            if re.search(r'\\b' + re.escape(word) + r'\\b', comm_lower):
                return FilterResult(passed=False, reason=f"Eval-sign contradiction: Blunder described with praise ('{word}')")

    is_good = played_best or (cp_loss is not None and cp_loss <= 0) or (mistake_type == "good")
    if is_good:
        for word in BLUNDER_WORDS:
            if re.search(r'\\b' + re.escape(word) + r'\\b', comm_lower):
                return FilterResult(passed=False, reason=f"Eval-sign contradiction: Good move described with blunder word ('{word}')")

    return FilterResult(passed=True)

PIECE_PATTERN = re.compile(r'\\b(queen|rook|bishop|knight|pawn|king)\\s+(?:on|at|to|from)\\s+([a-h][1-8])\\b', re.IGNORECASE)
NEGATIVE_ASSERTION_PREFIX = re.compile(r'\\b(?:absence\\s+of|lack\\s+of|without\\s+a|without\\s+any|without|no|neither)\\s+(?:a\\s+|an\\s+|the\\s+|any\\s+)?$', re.IGNORECASE)
PIECE_MAP = {
    'queen': chess.QUEEN, 'rook': chess.ROOK, 'bishop': chess.BISHOP,
    'knight': chess.KNIGHT, 'pawn': chess.PAWN, 'king': chess.KING,
}
UNICODE_HYPHENS = re.compile(r'[\\u2010\\u2011\\u2012\\u2013\\u2014\\u2015\\u2212]')
HYPHEN_MOVE_PATTERN = re.compile(r'(?<!\\w)(?:(?P<prefix>\\d+\\.+|\\.{2,3})\\s*)?(?P<san>[NBRQK]?[a-h][1-8][-–][a-h][1-8][+#]?)(?!\\w)', re.UNICODE)
STANDARD_SAN_PATTERN = re.compile(r'(?<!\\w)(?:(?P<prefix>\\d+\\.+|\\.{2,3})\\s*)?(?P<san>O-O(?:-O)?[+#]?|[NBRQK][a-h1-8]?x?[a-h][1-8](?:=[NBRQK])?[+#]?|[a-h]x[a-h][1-8](?:=[NBRQK])?[+#]?|[a-h][1-8]=[NBRQK][+#]?|[a-h][1-8][+#])(?!\\w)', re.UNICODE)
EXPLICIT_PAWN_PATTERN = re.compile(r'(?:\\b(?:move|plays|played)\\s+|(?:\\d+\\.+|\\.{2,3})\\s*)([a-h][1-8])\\b', re.IGNORECASE)

def extract_piece_square_mentions(commentary: str) -> List[Tuple[str, str, bool]]:
    mentions = []
    for m in PIECE_PATTERN.finditer(commentary):
        piece_name = m.group(1).lower()
        sq_name = m.group(2).lower()
        preceding = commentary[max(0, m.start() - 35):m.start()]
        is_neg = bool(NEGATIVE_ASSERTION_PREFIX.search(preceding))
        mentions.append((piece_name, sq_name, is_neg))
    return mentions

def extract_algebraic_move_references(commentary: str) -> List[Tuple[str, str]]:
    commentary = UNICODE_HYPHENS.sub('-', commentary)
    moves = []
    for m in HYPHEN_MOVE_PATTERN.finditer(commentary):
        san = m.group('san')
        raw = m.group(0)
        if (m.start() > 0 and commentary[m.start() - 1] == '-') or (m.end() < len(commentary) and commentary[m.end()] == '-'):
            continue
        following = commentary[m.end():m.end() + 20].lower()
        preceding = commentary[max(0, m.start() - 20):m.start()].lower()
        if (any(w in following for w in ["diag", "file", "rank", "line", "tension", "square"]) or
            any(w in preceding for w in ["along", "diagonal", "control", "cover", "guard", "defend", "tension", "between", "square"])):
            continue
        moves.append((san, raw, m.start()))

    for m in STANDARD_SAN_PATTERN.finditer(commentary):
        san = m.group('san')
        raw = m.group(0)
        moves.append((san, raw, m.start()))

    for m in EXPLICIT_PAWN_PATTERN.finditer(commentary):
        san = m.group(1)
        raw = m.group(0)
        moves.append((san, raw, m.start()))

    unique_moves = []
    seen_spans = []
    for san, raw, start in sorted(moves, key=lambda x: x[2]):
        end = start + len(raw)
        if any(not (end <= s or start >= e) for s, e in seen_spans):
            continue
        seen_spans.append((start, end))
        clean_san = san.strip().lstrip('.').rstrip('!?')
        unique_moves.append((clean_san, raw.strip()))
    return unique_moves

def is_legal_notation_on_board(b: chess.Board, notation: str) -> bool:
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
        if b.parse_san(notation) in b.legal_moves:
            return True
    except Exception:
        pass
    try:
        if chess.Move.from_uci(notation) in b.legal_moves:
            return True
    except Exception:
        pass
    if re.match(r'^[a-h][1-8]$', notation):
        sq = chess.parse_square(notation)
        if any(m.to_square == sq for m in b.legal_moves):
            return True
    return False

def is_destination_for_piece(b: chess.Board, expected_type: chess.PieceType, target_sq: chess.Square) -> bool:
    for m in b.legal_moves:
        if m.to_square == target_sq:
            p = b.piece_at(m.from_square)
            if p and (p.piece_type == expected_type or m.promotion == expected_type):
                return True
        if b.is_castling(m) and expected_type == chess.ROOK:
            castling_rook_dest = {chess.G1: chess.F1, chess.C1: chess.D1, chess.G8: chess.F8, chess.C8: chess.D8}.get(m.to_square)
            if castling_rook_dest == target_sq:
                return True
    return False

def is_move_legal_in_position_tree(board_before: chess.Board, played_move: Optional[chess.Move], best_move: Optional[chess.Move], notation: str) -> bool:
    if is_legal_notation_on_board(board_before, notation):
        return True
    b_opp = board_before.copy()
    b_opp.turn = not board_before.turn
    if is_legal_notation_on_board(b_opp, notation):
        return True
    if played_move and played_move in board_before.legal_moves:
        board_after = board_before.copy()
        board_after.push(played_move)
        if is_legal_notation_on_board(board_after, notation):
            return True
        for opp_m in list(board_after.legal_moves)[:30]:
            b_ply3 = board_after.copy()
            b_ply3.push(opp_m)
            if is_legal_notation_on_board(b_ply3, notation):
                return True
    if best_move and best_move in board_before.legal_moves:
        board_best = board_before.copy()
        board_best.push(best_move)
        if is_legal_notation_on_board(board_best, notation):
            return True
        for opp_m in list(board_best.legal_moves)[:30]:
            b_best_ply3 = board_best.copy()
            b_best_ply3.push(opp_m)
            if is_legal_notation_on_board(b_best_ply3, notation):
                return True
    return False

def is_piece_square_grounded_in_tree(board_before: chess.Board, played_move: Optional[chess.Move], best_move: Optional[chess.Move], expected_type: chess.PieceType, target_sq: chess.Square) -> bool:
    p = board_before.piece_at(target_sq)
    if p and p.piece_type == expected_type:
        return True
    if is_destination_for_piece(board_before, expected_type, target_sq):
        return True
    b_opp = board_before.copy()
    b_opp.turn = not board_before.turn
    if is_destination_for_piece(b_opp, expected_type, target_sq):
        return True
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

def validate_chess_grounding(fen: str, move_uci: str, commentary: str, best_move_uci: Optional[str] = None) -> FilterResult:
    try:
        board_before = chess.Board(fen)
        move = chess.Move.from_uci(move_uci) if move_uci else None
        best_move = chess.Move.from_uci(best_move_uci) if best_move_uci else None
    except Exception as e:
        return FilterResult(passed=False, reason=f"Invalid chess state: {e}")

    mentions = extract_piece_square_mentions(commentary)
    for piece_name, sq_name, is_negative in mentions:
        target_sq = chess.parse_square(sq_name)
        expected_type = PIECE_MAP.get(piece_name)
        if expected_type is None:
            continue
        if is_negative:
            p_current = board_before.piece_at(target_sq)
            if p_current and p_current.piece_type == expected_type:
                return FilterResult(passed=False, reason=f"Hallucination detected: Claimed absence of {piece_name} on {sq_name}, but {piece_name} is present")
        else:
            if not is_piece_square_grounded_in_tree(board_before, move, best_move, expected_type, target_sq):
                return FilterResult(passed=False, reason=f"Hallucination detected: No {piece_name} on or moving to square {sq_name}")

    move_refs = extract_algebraic_move_references(commentary)
    for clean_san, raw_token in move_refs:
        if not is_move_legal_in_position_tree(board_before, move, best_move, clean_san):
            return FilterResult(passed=False, reason=f"Illegal move mentioned in commentary: '{raw_token}' is not legal in this position tree")

    return FilterResult(passed=True)

# NLP Metric functions
def _tokenize(text: str) -> List[str]:
    return re.findall(r'\\b\\w+\\b', text.lower())

def _lcs(x: List[str], y: List[str]) -> int:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[m][n]

def compute_rouge_l_single(pred: str, ref: str) -> float:
    p_tokens = _tokenize(pred)
    r_tokens = _tokenize(ref)
    if not p_tokens or not r_tokens:
        return 0.0
    lcs_len = _lcs(p_tokens, r_tokens)
    prec = lcs_len / len(p_tokens)
    rec = lcs_len / len(r_tokens)
    if prec + rec == 0:
        return 0.0
    return round((2 * prec * rec) / (prec + rec), 4)

def compute_token_f1_single(pred: str, ref: str) -> float:
    p_set = set(_tokenize(pred))
    r_set = set(_tokenize(ref))
    if not p_set or not r_set:
        return 0.0
    common = len(p_set & r_set)
    if common == 0:
        return 0.0
    prec = common / len(p_set)
    rec = common / len(r_set)
    return round((2 * prec * rec) / (prec + rec), 4)
"""

    cell_3_code = f"""# 4. Load Stage 5 Test Data and Execute Inference
import base64, gzip, json, time
from collections import defaultdict

STAGE5_DATA_B64 = "{b64_str}"
test_items = json.loads(gzip.decompress(base64.b64decode(STAGE5_DATA_B64)).decode("utf-8"))
print(f"Loaded {{len(test_items)}} test positions from payload.")

eos_ids = [tokenizer.eos_token_id]
end_id = tokenizer.convert_tokens_to_ids('<|end|>')
if end_id is not None and end_id not in eos_ids:
    eos_ids.append(end_id)
print(f"Configured eos_token_ids: {{eos_ids}}")

results = []
start_time = time.time()
print(f"Starting generation for {{len(test_items)}} items with greedy decoding (max_new_tokens=85)...\\n")

for i, item in enumerate(test_items):
    prompt = item["full_serving_prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=85,
            do_sample=False,
            temperature=None,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_tokens = outputs[0][prompt_len:]
    commentary = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # Scaffolding check
    scaff_res = validate_no_scaffolding_leakage(commentary)

    # Chess Grounding check
    ground_res = validate_chess_grounding(
        fen=item["fen"],
        move_uci=item["move_uci"],
        commentary=commentary,
        best_move_uci=item.get("best_move")
    )

    # Eval Sentiment Agreement check
    loss = item.get("cp_loss")
    if loss is None:
        loss = 300 if item.get("mistake_type") == "blunder" else 0
    played_best = (item.get("mistake_type") == "good")
    eval_res = validate_eval_sign_consistency(
        commentary=commentary,
        cp_loss=loss,
        played_best=played_best,
        mistake_type=item.get("mistake_type", "good")
    )

    # Tactic Recall check
    motif = item.get("motif", "none")
    is_tactical = motif not in ("none", "unknown", "opening_principle", "endgame_technique")
    tactic_recalled = False
    if is_tactical:
        tactic_clean = motif.lower().replace("_", " ")
        base_words = [w for w in tactic_clean.split() if len(w) > 3]
        comm_lower = commentary.lower()
        tactic_recalled = any(w in comm_lower for w in base_words)

    # NLP text metrics
    ref = item.get("reference_commentary", "")
    r_l = compute_rouge_l_single(commentary, ref)
    f1 = compute_token_f1_single(commentary, ref)

    record = dict(item)
    record.update({{
        "generated_commentary": commentary,
        "grounding_passed": ground_res.passed,
        "grounding_reason": ground_res.reason,
        "scaffolding_passed": scaff_res.passed,
        "scaffolding_reason": scaff_res.reason,
        "eval_sentiment_passed": eval_res.passed,
        "eval_sentiment_reason": eval_res.reason,
        "is_tactical": is_tactical,
        "tactic_recalled": tactic_recalled if is_tactical else None,
        "rouge_l": r_l,
        "token_f1": f1,
    }})
    results.append(record)

    if (i + 1) % 25 == 0 or (i + 1) == len(test_items):
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        rem = (len(test_items) - (i + 1)) / rate if rate > 0 else 0
        print(f"[{{i+1:3d}}/{{len(test_items)}}] Elapsed: {{elapsed:.1f}}s | Rate: {{rate:.2f}} ex/s | Rem: {{rem:.1f}}s | Latest: [{{record['motif']}}] {{record['move_san']}} -> Grounding: {{ground_res.passed}} | Leakage: {{not scaff_res.passed}}")

total_time = time.time() - start_time
print(f"\\nGeneration & evaluation finished in {{total_time:.2f}}s (avg {{total_time/len(test_items):.2f}}s/ex).\\n")

# Compile Summary
total = len(results)
overall_rouge = sum(r["rouge_l"] for r in results) / total
overall_f1 = sum(r["token_f1"] for r in results) / total
overall_eval = sum(1 for r in results if r["eval_sentiment_passed"]) / total
tactical_records = [r for r in results if r["is_tactical"]]
overall_recall = sum(1 for r in tactical_records if r["tactic_recalled"]) / len(tactical_records) if tactical_records else 1.0
overall_grounding = sum(1 for r in results if r["grounding_passed"]) / total
overall_hallucination = 1.0 - overall_grounding
overall_scaff_leaks = sum(1 for r in results if not r["scaffolding_passed"])

# Per-category breakdown
cat_groups = defaultdict(list)
for r in results:
    cat_groups[r["motif"]].append(r)

per_cat_summary = {{}}
for motif, recs in sorted(cat_groups.items(), key=lambda x: -len(x[1])):
    n = len(recs)
    r_l = sum(r["rouge_l"] for r in recs) / n
    f1 = sum(r["token_f1"] for r in recs) / n
    e_pass = sum(1 for r in recs if r["eval_sentiment_passed"])
    e_rate = e_pass / n
    tactical_in_cat = [r for r in recs if r["is_tactical"]]
    t_pass = sum(1 for r in tactical_in_cat if r["tactic_recalled"]) if tactical_in_cat else None
    t_rate = (t_pass / len(tactical_in_cat)) if tactical_in_cat else None
    g_pass = sum(1 for r in recs if r["grounding_passed"])
    g_rate = g_pass / n
    h_count = n - g_pass
    h_rate = h_count / n
    s_leak = sum(1 for r in recs if not r["scaffolding_passed"])
    s_rate = s_leak / n

    per_cat_summary[motif] = {{
        "n": n,
        "rouge_l": round(r_l, 4),
        "token_f1": round(f1, 4),
        "eval_sentiment_passed": e_pass,
        "eval_sentiment_rate": round(e_rate, 4),
        "tactic_recalled": t_pass,
        "tactic_recall_rate": round(t_rate, 4) if t_rate is not None else None,
        "grounding_passed": g_pass,
        "grounding_rate": round(g_rate, 4),
        "hallucination_count": h_count,
        "hallucination_rate": round(h_rate, 4),
        "scaffolding_leaks": s_leak,
        "scaffolding_leak_rate": round(s_rate, 4),
    }}

summary_output = {{
    "total_samples": total,
    "overall": {{
        "rouge_l": round(overall_rouge, 4),
        "token_f1": round(overall_f1, 4),
        "eval_sentiment_agreement": round(overall_eval, 4),
        "tactic_recall": round(overall_recall, 4),
        "chess_grounding_pass_rate": round(overall_grounding, 4),
        "hallucination_rate": round(overall_hallucination, 4),
        "scaffolding_leakage_rate": round(overall_scaff_leaks / total, 4),
        "scaffolding_leakage_count": overall_scaff_leaks,
    }},
    "per_category": per_cat_summary,
}}

# Print Category Breakdown Table
print("=" * 115)
print(f"{{'MOTIF (CATEGORY)':<25}} | {{'N':<4}} | {{'ROUGE-L':<8}} | {{'F1':<6}} | {{'SENTIMENT':<15}} | {{'TACTIC REC':<15}} | {{'GROUNDING (PASS)':<18}} | {{'LEAKS':<7}}")
print("-" * 115)
for motif, stats in per_cat_summary.items():
    n = stats["n"]
    rl = stats["rouge_l"]
    f1 = stats["token_f1"]
    sent = f"{{stats['eval_sentiment_passed']}}/{{n}} ({{stats['eval_sentiment_rate']*100:.1f}}%)"
    trec = f"{{stats['tactic_recalled']}}/{{n}} ({{stats['tactic_recall_rate']*100:.1f}}%)" if stats["tactic_recalled"] is not None else "N/A"
    grnd = f"{{stats['grounding_passed']}}/{{n}} ({{stats['grounding_rate']*100:.1f}}%)"
    leaks = f"{{stats['scaffolding_leaks']}}/{{n}}"
    print(f"{{motif:<25}} | {{n:<4}} | {{rl:<8.4f}} | {{f1:<6.4f}} | {{sent:<15}} | {{trec:<15}} | {{grnd:<18}} | {{leaks:<7}}")

print("=" * 115)
print(f"\\nOVERALL METRICS (N={{total}}):")
print(f"  ROUGE-L:                 {{summary_output['overall']['rouge_l']}}")
print(f"  Token-F1:                {{summary_output['overall']['token_f1']}}")
print(f"  Eval-Sentiment Agree:    {{summary_output['overall']['eval_sentiment_agreement']*100:.2f}}%")
print(f"  Tactic Recall:           {{summary_output['overall']['tactic_recall']*100:.2f}}%")
print(f"  Grounding Pass Rate:     {{summary_output['overall']['chess_grounding_pass_rate']*100:.2f}}%")
print(f"  Hallucination Rate:      {{summary_output['overall']['hallucination_rate']*100:.2f}}%")
print(f"  Scaffolding Leaks:       {{summary_output['overall']['scaffolding_leakage_count']}}/{{total}} ({{summary_output['overall']['scaffolding_leakage_rate']*100:.2f}}%)")
print("=" * 115)

with open("stage5_predictions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

with open("stage5_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_output, f, indent=2)

print("Saved stage5_predictions.json and stage5_summary.json successfully!")
"""

    notebook = {
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_0_code]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_1_code]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_2_code]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell_3_code]},
        ],
        "metadata": {
            "language_info": {"name": "python"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    out_path = Path("notebooks_stage5_eval/stage5_eval.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Written Stage 5 evaluation notebook to {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    build_notebook()
