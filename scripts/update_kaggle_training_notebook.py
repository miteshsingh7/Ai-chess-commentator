"""Package resynced dataset and source code into kaggle_qlora_training.ipynb."""

import io
import tarfile
import base64
import json
import os

def package_and_update():
    # 1. Create tarball
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for fname in [
            "data/commentary_sft_train.jsonl",
            "data/commentary_sft_val.jsonl",
            "data/commentary_sft_test.jsonl",
            "data/smoke_test_v8_inputs.json",
        ]:
            tar.add(fname)
        for root, dirs, files in os.walk("src"):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if not f.endswith(".pyc"):
                    fpath = os.path.join(root, f)
                    tar.add(fpath)

    buf.seek(0)
    raw_bytes = buf.read()
    b64_str = base64.b64encode(raw_bytes).decode("utf-8")
    print(f"Archive size: {len(raw_bytes)} bytes ({len(raw_bytes)/1024:.1f} KB)")
    print(f"Base64 length: {len(b64_str)} chars")

    # 2. Update notebook
    nb_path = "notebooks/kaggle_qlora_training.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 2: Unpack
    cell_2_code = [
        "# Step 2: Unpack Embedded Dataset & Source Code\n",
        "import base64, io, tarfile, os, sys\n",
        "\n",
        f'B64_DATA = "{b64_str}"\n',
        "\n",
        'print("Unpacking datasets and src package...")\n',
        "raw = base64.b64decode(B64_DATA)\n",
        'with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:\n',
        '    tar.extractall(".")\n',
        'print("Unpacked successfully!")\n',
        'if "src" not in sys.path:\n',
        '    sys.path.insert(0, "src")\n',
        'print("Python path configured.")\n',
    ]
    nb["cells"][2]["source"] = cell_2_code

    # Cell 6: Post-Training Smoke Test & Validation
    cell_6_code = [
        "# Step 6: Post-Training Smoke Test Across Diverse Categories\n",
        "import json, os, shutil\n",
        "import pandas as pd\n",
        "import torch\n",
        "from chess_commentator.serving.engine import CommentaryInferenceEngine\n",
        "from chess_commentator.dataset.checker import validate_chess_grounding\n",
        "from chess_commentator.dataset.filters import validate_no_scaffolding_leakage\n",
        "\n",
        'print("Loading fine-tuned model for smoke test...")\n',
        "engine = CommentaryInferenceEngine(\n",
        "    base_model_id=config.model_id,\n",
        "    adapter_path=config.output_dir,\n",
        "    load_in_4bit=True,\n",
        ")\n",
        "engine.load()\n",
        "\n",
        'with open("data/smoke_test_v8_inputs.json") as f:\n',
        "    smoke_positions = json.load(f)\n",
        "\n",
        "results = []\n",
        "for p in smoke_positions:\n",
        '    prompt = p["full_serving_prompt"]\n',
        "    comm = engine.generate(prompt)\n",
        "    chk_grounding = validate_chess_grounding(\n",
        '        fen=p["fen"],\n',
        '        move_uci=p["move_uci"],\n',
        "        commentary=comm,\n",
        '        best_move_uci=p.get("best_move"),\n',
        "    )\n",
        "    chk_leakage = validate_no_scaffolding_leakage(comm)\n",
        "    p_res = dict(p)\n",
        '    p_res["generated_commentary"] = comm\n',
        '    p_res["grounding_passed"] = chk_grounding.passed\n',
        '    p_res["grounding_reason"] = chk_grounding.reason\n',
        '    p_res["leakage_passed"] = chk_leakage.passed\n',
        '    p_res["leakage_reason"] = chk_leakage.reason\n',
        "    results.append(p_res)\n",
        '    print("-" * 50)\n',
        '    print(f"[{p[\'category\']}] {p[\'move_san\']} ({p[\'move_uci\']})")\n',
        '    print(f"Commentary: {comm}")\n',
        '    print(f"Scaffolding Leak: {\'NO\' if chk_leakage.passed else \'YES: \' + chk_leakage.reason}")\n',
        '    print(f"Grounding Passed: {\'YES\' if chk_grounding.passed else \'NO: \' + chk_grounding.reason}")\n',
        "\n",
        'out_json_path = "/kaggle/working/smoke_test_results.json"\n',
        'with open(out_json_path, "w") as f:\n',
        "    json.dump(results, f, indent=2)\n",
        'print(f"Saved {out_json_path}")\n',
        "\n",
        'clean_adapter_dir = "/kaggle/working/phi35_adapter_clean"\n',
        "os.makedirs(clean_adapter_dir, exist_ok=True)\n",
        "for item in os.listdir(config.output_dir):\n",
        "    s = os.path.join(config.output_dir, item)\n",
        "    d = os.path.join(clean_adapter_dir, item)\n",
        "    if os.path.isfile(s):\n",
        "        shutil.copy2(s, d)\n",
        'with open(os.path.join(clean_adapter_dir, "dataset-metadata.json"), "w") as f:\n',
        '    json.dump({"title": "phi35-chess-adapter-v10", "id": "miteshsingh7/phi35-chess-adapter-v10", "licenses": [{"name": "MIT"}]}, f, indent=2)\n',
        'print(f"Prepared clean adapter directory at {clean_adapter_dir}")\n',
        "\n",
        "df_summary = pd.DataFrame([{\n",
        '    "Category": r["category"],\n',
        '    "Move": r["move_san"],\n',
        '    "Leakage Clean": "YES" if r["leakage_passed"] else "LEAK",\n',
        '    "Grounding": "PASS" if r["grounding_passed"] else "FAIL",\n',
        '    "Reason": r["grounding_reason"],\n',
        "} for r in results])\n",
        'print("\\n" + "=" * 60)\n',
        'print("SMOKE TEST SUMMARY EVALUATION TABLE")\n',
        'print("=" * 60)\n',
        "print(df_summary.to_string(index=False))\n",
        'total_clean = sum(r["leakage_passed"] for r in results)\n',
        'total_grounding = sum(r["grounding_passed"] for r in results)\n',
        'print(f"\\nScaffolding Clean Rate: {total_clean} / {len(results)}")\n',
        'print(f"Grounding Pass Rate:    {total_grounding} / {len(results)}")\n',
    ]
    nb["cells"][6]["source"] = cell_6_code

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully updated {nb_path}!")


if __name__ == "__main__":
    package_and_update()
