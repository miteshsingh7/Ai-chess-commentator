# ♟️ AI Chess Commentator

Fine-tuning small open Language Models (**Phi-3.5-mini-instruct** / **Llama-3.1-8B-Instruct**) to generate natural-language chess commentary explaining why moves are good or bad, strictly grounded in **Stockfish evaluations** and **15+ tactical blunder taxonomies** adapted from [chessIQ](https://github.com/miteshsingh7/chessIQ).

---

## 🚀 Key Features

- **Domain-Grounded Analysis**: Adapts chessIQ's phase 1–4 modules into a standalone engine stripped of Streamlit/UI code.
- **15+ Tactical Taxonomies**: Classifies forks (knight, pawn, piece), pins, skewers, discovered attacks, back-rank mate, hanging pieces, trapped pieces, overloaded pieces, zwischenzugs, and phase technique errors.
- **Teacher LLM Dataset Generation**: Generates master-level 2–4 sentence commentaries using **Groq API** (`qwen/qwen3.8-27b`) with a SHA-256 content-hash disk cache to avoid duplicate API calls.
- **Strict Quality Filtering**: Filters out eval-sign contradictions (e.g. praising blunders), hallucinated pieces/squares, and generic robotic filler.
- **Kaggle GPU Ready (QLoRA)**: 4-bit `bitsandbytes` + `peft` + `trl` training pipeline runnable on a single Kaggle GPU (16GB T4 or 40GB A100).
- **Dual Evaluation**: Evaluates models using automated metrics (ROUGE-L, Token-F1, Eval Agreement, Tactic Recall, Hallucination Rate) and an LLM-as-a-Judge 1–5 scoring rubric.
- **Live Serving**: Single-position Python API `generate_commentary(fen, move)` and interactive CLI demo.

---

## 📂 Repository Structure

```text
ai-chess-commentator/
├── configs/
│   ├── train_phi35.yaml             # QLoRA config for Phi-3.5-mini (MIT License)
│   └── train_llama31.yaml           # QLoRA config for Llama-3.1-8B
├── notebooks/
│   └── kaggle_qlora_training.ipynb  # 1-Click Kaggle GPU notebook
├── scripts/
│   ├── demo_cli.py                  # Live interactive commentary CLI
│   ├── run_analysis.py              # PGN batch analysis & taxonomy classifier
│   ├── run_teacher_generation.py    # Teacher generation with Groq API
│   ├── run_dataset_cleaning.py      # Quality filters, balancing, and ChatML export
│   ├── train_qlora.py               # Standalone training script
│   └── run_eval.py                  # Benchmark & evaluation report
├── src/
│   └── chess_commentator/
│       ├── analysis/                # Stage 1: Adapted from chessIQ
│       │   ├── constants.py         # Piece values, CP_CAP, thresholds
│       │   ├── models.py            # Frozen dataclasses (PositionAnalysis, etc.)
│       │   ├── engine.py            # Stockfish wrapper (depth-18, fast/deep)
│       │   ├── features.py          # Positional & tactical feature extractor
│       │   ├── taxonomy.py          # 15+ tactical taxonomy detectors
│       │   ├── parser.py            # PGN move sequence parser
│       │   ├── fetcher.py           # Chess.com game downloader
│       │   └── pipeline.py          # Unified pipeline & single-move analyzer
│       ├── teacher/                 # Stage 2: Teacher LLM generation
│       │   ├── prompt_builder.py    # Grounded prompt construction
│       │   ├── client.py            # Groq API client with sliding-window rate limiter & mock mode
│       │   ├── cache.py             # SHA-256 content-hash disk cache
│       │   └── generator.py         # Batch generation orchestrator
│       ├── dataset/                 # Stage 3: Quality curation
│       │   ├── filters.py           # Eval-sign and filler filters
│       │   ├── checker.py           # Chess rule & hallucination checker
│       │   ├── balancer.py          # Deduplication & game-aware split
│       │   └── formatter.py         # ChatML JSONL formatter
│       ├── training/                # Stage 4: QLoRA Fine-tuning
│       │   ├── config.py            # TrainingConfig dataclass
│       │   ├── model_loader.py      # 4-bit NF4 quantization & LoRA config
│       │   └── trainer.py           # TRL SFTTrainer training loop
│       ├── evaluation/              # Stage 5: Evaluation suite
│       │   ├── metrics.py           # ROUGE-L, F1, sentiment agreement, hallucination
│       │   ├── judge.py             # LLM-as-a-judge 1-5 rubric scoring
│       │   └── benchmark.py         # Benchmark runner
│       └── serving/                 # Stage 6: Live inference
│           ├── engine.py            # CommentaryInferenceEngine
│           ├── service.py           # generate_commentary(fen, move) entrypoint
│           └── api.py               # FastAPI REST service
└── tests/                           # Comprehensive 36+ test suite
```

---

## ⚡ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/miteshsingh7/chess-commentator.git
cd ai-chess-commentator

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

Ensure Stockfish is installed on your system:
- **macOS**: `brew install stockfish`
- **Linux / Ubuntu**: `apt-get install stockfish`
- **Windows**: Download binary from [StockfishChess.org](https://stockfishchess.org/download/)

### 2. Live Commentary Demo

Test commentary on any position in your terminal:

```bash
# Analyze Scholar's mate (winning checkmate)
python scripts/demo_cli.py --fen "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4" --move "Qxf7#"

# Analyze a hanging knight blunder
python scripts/demo_cli.py --fen "r1bqkbnr/ppp1pppp/3p4/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 3" --move "f3e5"
```

### 3. Python API Usage

```python
from chess_commentator.serving.service import generate_commentary

result = generate_commentary(
    fen="r1bqkbnr/ppp1pppp/3p4/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 3",
    move="f3e5",
    depth=18,
)

print("Played Move:", result.move)
print("Tactical Motif:", result.taxonomy)
print("Centipawn Loss:", result.cp_loss)
print("Blunder Flag:", result.is_blunder)
print("Commentary:", result.commentary)
```

---

## 🧠 Model Weights & Verification Guide (For Reviewers)

### Where to Access the Trained Model
The fine-tuned QLoRA weights (`v10`) are publicly hosted and accessible via:
1. **Hugging Face Hub**: [`Mitex07/phi35-chess-adapter-v10`](https://huggingface.co/Mitex07/phi35-chess-adapter-v10)
   * **Base Model**: [`microsoft/Phi-3.5-mini-instruct`](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) (3.8B parameters)
   * **Adapter Architecture**: QLoRA (`r=16`, `lora_alpha=32`, `lora_dropout=0.05`, all-linear projection targets)
2. **Kaggle Dataset**: [`miteshsingh7/phi35-chess-adapter-v10`](https://www.kaggle.com/datasets/miteshsingh7/phi35-chess-adapter-v10)
   * Trained over 3 epochs on Kaggle T4 GPU using 2,499 verification-augmented master positions.

### Hardware & Local Evaluation Note
* **Full Local Weights (Inference)**: Loading the full 3.8B unquantized base model (`microsoft/Phi-3.5-mini-instruct`) + LoRA adapter in 16-bit precision requires **~7.6 GB RAM / VRAM**.
  * On a workstation or GPU instance with $\ge 12\text{ GB}$ VRAM (or $\ge 16\text{ GB}$ system RAM), the adapter can be loaded directly using `PeftModel`.
  * On memory-constrained local machines (e.g. laptops with $\le 8\text{ GB}$ RAM), running a 3.8B model in float16 causes OS disk paging / swapping.
  * For this reason, the serving engine (`src/chess_commentator/serving/engine.py`) provides a multi-tiered architecture:
    1. **Local Weights**: Loads `PeftModel` when sufficient RAM/VRAM is available.
    2. **Groq Cloud LLM Integration**: Uses Groq (`qwen/qwen3.8-27b`) when `GROQ_API_KEY` is present, delivering live Grandmaster commentary in $<1.0\text{s}$ with zero local memory footprint.
    3. **Dynamic Rule-Based Engine**: Position-aware tactical and strategic analyzer that produces detailed 2–3 sentence commentary when offline.

### Loading the Model in Python

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "microsoft/Phi-3.5-mini-instruct"
adapter_id = "Mitex07/phi35-chess-adapter-v10"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()
```

---

## 🎨 Interactive Frontend — Chess Analysis Salon

Launch the local development server (serves the interactive UI at `/` and the REST API at `/v1/commentary`):

```bash
python scripts/serve_app.py
```
Open **`http://localhost:8001`** in your browser.

- **Interactive Move Stepping**: Step forward and backward through games using arrow keys (`←`/`→`) or playback controls.
- **PGN Game Loading**: Load famous master games (e.g. *Morphy Opera Game 1858*) or paste custom PGN notation.
- **Real-Time Evaluation Bar**: Live centipawn win-probability indicator powered by Stockfish.
- **AI Commentary Insight Card**: Displays move quality badges (Best Move, Good Move, Inaccuracy, Mistake, Blunder), tactical motif pills, and live natural-language commentary.
- **Board Utilities**: Flip board (`F` key or button) and manual position analysis.

---

## 🔁 End-to-End Pipeline Workflow

### Stage 1: PGN Analysis & Feature Extraction
```bash
python scripts/run_analysis.py --pgn data/raw_pgn/sample.pgn --output data/processed/moves_analyzed.parquet --depth 18
```

### Stage 2: Teacher Commentary Generation (Groq API)
```bash
export GROQ_API_KEY="your_api_key"
python scripts/run_teacher_generation.py --input data/processed/moves_analyzed.parquet --output data/processed/moves_with_commentary.parquet --model qwen/qwen3.8-27b
```
*Generated responses are cached in `.cache/teacher_commentary/` by SHA-256 hash to eliminate redundant API spend.*

### Stage 3: Quality Filtering & Dataset Balancing
```bash
python scripts/run_dataset_cleaning.py --input data/processed/moves_with_commentary.parquet --output-dir data/dataset/
```
Outputs `train.jsonl`, `val.jsonl`, and `test.jsonl` in ChatML format.

### Stage 4: Fine-Tuning on Kaggle GPU
Upload `notebooks/kaggle_qlora_training.ipynb` to Kaggle, enable GPU T4 or A100, and run the notebook to train in 4-bit QLoRA.
Or run locally:
```bash
python scripts/train_qlora.py --config configs/train_phi35.yaml
```

### Stage 5: Evaluation
```bash
python scripts/run_eval.py --test-data data/dataset/test.jsonl --mock-judge
```

---

## 🧪 Testing

Run the full pytest suite:
```bash
pytest tests/ -v
```

All 36 unit and integration tests verify:
- Feature extraction & pawn structure metrics
- 15+ tactical taxonomy detectors (fork, pin, skewer, hanging, trapped, etc.)
- Stockfish evaluation and centipawn loss calculations
- Teacher prompt grounding and SHA-256 content-hash caching
- Eval-sign consistency and chess hallucination checkers
- QLoRA training configurations and device resolution
- NLP metrics (ROUGE-L, F1, Tactic recall, Hallucination rate) and LLM judge
- Live serving API and CLI contracts

---

## 📜 License
MIT License
