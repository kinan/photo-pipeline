# scripts/compare_models.py
# Runs qwen2.5-vl-7b-4bit and qwen2.5-vl-3b-4bit on all photos via mlx_vlm,
# saves both sets of results to the model_inferences table, then exits.
#
# Run: uv run python scripts/compare_models.py

import json, sys, time
from pathlib import Path
import duckdb
from mlx_vlm import load, generate, convert
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

sys.path.insert(0, str(Path(__file__).parent))
from process_photo import (
    OBSERVER_PROMPT, SEO_PROMPT,
    file_hash, parse_vlm_json, clean_seo,
)

QWEN7B_HF   = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
QWEN3B_HF   = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
QWEN7B_PATH = "./models/qwen2.5-vl-7b-4bit"
QWEN3B_PATH = "./models/qwen2.5-vl-3b-4bit"
DB_PATH     = "./outputs/photos.duckdb"
RAW_DIR     = Path("./data/raw")

MODELS = [
    ("qwen2.5-vl-7b-4bit", QWEN7B_PATH, QWEN7B_HF),
    ("qwen2.5-vl-3b-4bit", QWEN3B_PATH, QWEN3B_HF),
]


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def ensure_table(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS model_inferences (
            photo_id         VARCHAR,
            model_name       VARCHAR,
            observation      VARCHAR,
            name             VARCHAR,
            description      VARCHAR,
            caption          VARCHAR,
            keywords         VARCHAR,
            content_location VARCHAR,
            slug             VARCHAR,
            vlm_latency_ms   INTEGER,
            PRIMARY KEY (photo_id, model_name)
        )
    """)


def already_done(con, photo_id: str, model_name: str) -> bool:
    return con.execute(
        "SELECT 1 FROM model_inferences WHERE photo_id = ? AND model_name = ?",
        [photo_id, model_name]
    ).fetchone() is not None


def save_result(con, photo_id: str, model_name: str, observation: str, seo: dict, latency_ms: int):
    con.execute("""
        INSERT OR REPLACE INTO model_inferences
            (photo_id, model_name, observation, name, description, caption,
             keywords, content_location, slug, vlm_latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        photo_id, model_name, observation,
        seo.get("name"), seo.get("description"), seo.get("caption"),
        json.dumps(seo.get("keywords", [])),
        seo.get("contentLocation"), seo.get("slug"),
        latency_ms,
    ])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_vlm(model, processor, config, image_path: str, prompt: str, max_tokens: int) -> tuple[str, int]:
    formatted = apply_chat_template(processor, config, prompt, num_images=1)
    t0 = time.time()
    output = generate(model, processor, formatted, image_path, max_tokens=max_tokens, verbose=False)
    return output.text.strip(), int((time.time() - t0) * 1000)


def infer_photo(model, processor, config, image_path: str) -> tuple[str, dict, int]:
    observation, t1 = run_vlm(model, processor, config, image_path, OBSERVER_PROMPT, 300)
    seo_raw, t2 = run_vlm(
        model, processor, config, image_path,
        SEO_PROMPT.replace("{observation}", observation), 600
    )
    try:
        seo = clean_seo(parse_vlm_json(seo_raw, label="seo"))
    except json.JSONDecodeError:
        seo = {}
    return observation, seo, t1 + t2


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def ensure_model(model_path: str, hf_id: str):
    path = Path(model_path)
    if path.exists() and any(path.glob("*.safetensors")):
        print(f"  Found local model at {model_path}")
        return
    print(f"  Downloading {hf_id} → {model_path} ...")
    convert(hf_id, model_path, quantize=True, q_bits=4)
    print(f"  Saved.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    photos = sorted(RAW_DIR.glob("*.JPG")) + sorted(RAW_DIR.glob("*.jpg"))
    if not photos:
        print(f"No photos found in {RAW_DIR}")
        sys.exit(1)

    con = duckdb.connect(DB_PATH)
    ensure_table(con)

    for model_name, model_path, hf_id in MODELS:
        print(f"\n{'='*60}")
        print(f"[{model_name}]")
        ensure_model(model_path, hf_id)

        print(f"  Loading model...")
        model, processor = load(model_path)
        config = load_config(model_path)
        print(f"  Model loaded. Running on {len(photos)} photo(s)...\n")

        for i, photo in enumerate(photos, 1):
            path = str(photo.resolve())
            photo_id = file_hash(path)

            if already_done(con, photo_id, model_name):
                print(f"  [{i}/{len(photos)}] {photo.name} — already done, skipping")
                continue

            print(f"  [{i}/{len(photos)}] {photo.name} [{photo_id}]")
            observation, seo, latency = infer_photo(model, processor, config, path)
            save_result(con, photo_id, model_name, observation, seo, latency)
            print(f"    name:    {seo.get('name')}")
            print(f"    latency: {latency}ms")

        del model, processor, config

    con.close()
    print(f"\n{'='*60}")
    print("All done. Run `uv run python scripts/analyze_comparison.py` to see the analysis.")


if __name__ == "__main__":
    main()
