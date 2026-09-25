import json
from chess_commentator.dataset.formatter import construct_user_prompt, SYSTEM_INSTRUCTION
from chess_commentator.analysis.engine import StockfishEngine

positions = json.load(open('data/smoke_test_v7_inputs.json'))
with StockfishEngine(default_depth=10) as sf:
    for p in positions:
        cp_str = p['engine'].split('loss: ')[1].split(' cp')[0] if 'loss: ' in p['engine'] else '0'
        cp_loss = int(cp_str)
        m_type = p['engine'].split()[0].lower()
        played_best = (p['best_move'] == p['move_uci'])
        u_text = construct_user_prompt(
            fen=p['fen'],
            move_san=p['move_san'],
            move_uci=p['move_uci'],
            player_color=p['color'].lower(),
            mistake_type=m_type,
            cp_loss=cp_loss,
            best_move=p['best_move'],
            played_best=played_best,
            tactic_type=p['motif'],
            mistake_category=p['category'],
            stockfish_engine=sf,
        )
        p['full_serving_prompt'] = f"<|system|>\n{SYSTEM_INSTRUCTION}<|end|>\n<|user|>\n{u_text}<|end|>\n<|assistant|>\n"
        print("-------------------------------------------")
        print(f"[{p['category']}]")
        print(p['full_serving_prompt'])

with open('data/smoke_test_v8_inputs.json', 'w') as f:
    json.dump(positions, f, indent=2)
print("Saved data/smoke_test_v8_inputs.json")
