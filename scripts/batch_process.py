# scripts/batch_process.py
# Loads the VLM once and processes all photos in data/raw.
# Skips photos already in the database. Run with: uv run python scripts/batch_process.py

import json, hashlib, subprocess, sys, time, re
from pathlib import Path
import duckdb
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Import prompts and helpers from process_photo
sys.path.insert(0, str(Path(__file__).parent))
from process_photo import (
    OBSERVER_PROMPT, SEO_PROMPT,
    file_hash, extract_exif, parse_vlm_json, clean_seo, build_schema_org
)

MODEL_PATH = "./models/qwen2.5-vl-7b-4bit"
DB_PATH    = "./outputs/photos.duckdb"
RAW_DIR    = Path("./data/raw")


def run_vlm(model, processor, config, image_path: str, prompt: str, max_tokens: int = 400) -> tuple[str, int]:
    formatted = apply_chat_template(processor, config, prompt, num_images=1)
    t0 = time.time()
    output = generate(model, processor, formatted, image_path, max_tokens=max_tokens, verbose=False)
    latency_ms = int((time.time() - t0) * 1000)
    return output.text.strip(), latency_ms


def ensure_table(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            photo_id            VARCHAR PRIMARY KEY,
            original_path       VARCHAR,
            pipeline_state      VARCHAR,
            camera_make         VARCHAR,
            camera_model        VARCHAR,
            lens_model          VARCHAR,
            iso                 INTEGER,
            shutter_speed       DOUBLE,
            aperture            DOUBLE,
            focal_length_mm     DOUBLE,
            capture_date        VARCHAR,
            gps_lat             DOUBLE,
            gps_lon             DOUBLE,
            observation         VARCHAR,
            name                VARCHAR,
            description         VARCHAR,
            caption             VARCHAR,
            keywords            VARCHAR,
            content_location    VARCHAR,
            slug                VARCHAR,
            schema_org          VARCHAR,
            vlm_latency_ms      INTEGER
        )
    """)


def already_processed(con, photo_id: str) -> bool:
    result = con.execute("SELECT 1 FROM photos WHERE photo_id = ?", [photo_id]).fetchone()
    return result is not None


def process_one(model, processor, config, con, image_path: Path, index: int, total: int):
    path = str(image_path.resolve())
    photo_id = file_hash(path)

    if already_processed(con, photo_id):
        print(f"[{index}/{total}] Skipping {image_path.name} (already processed)")
        return

    print(f"\n[{index}/{total}] {image_path.name} [{photo_id}]")

    exif = extract_exif(path)

    print(f"  Step 1/2: Observing...")
    observation, t1 = run_vlm(model, processor, config, path, OBSERVER_PROMPT, max_tokens=300)

    print(f"  Step 2/2: SEO metadata...")
    seo_raw, t2 = run_vlm(
        model, processor, config, path,
        SEO_PROMPT.replace("{observation}", observation),
        max_tokens=600
    )
    try:
        seo = parse_vlm_json(seo_raw, label="seo")
        seo = clean_seo(seo)
    except json.JSONDecodeError:
        seo = {}

    total_latency = t1 + t2
    schema_org = build_schema_org(seo, exif, photo_id, path)

    con.execute("""
        INSERT OR REPLACE INTO photos (
            photo_id, original_path, pipeline_state,
            camera_make, camera_model, lens_model,
            iso, shutter_speed, aperture, focal_length_mm,
            capture_date, gps_lat, gps_lon,
            observation,
            name, description, caption, keywords, content_location, slug,
            schema_org, vlm_latency_ms
        ) VALUES (?, ?, 'inferred', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        photo_id, path,
        exif["camera_make"], exif["camera_model"], exif["lens_model"],
        exif["iso"], exif["shutter_speed"], exif["aperture"], exif["focal_length_mm"],
        exif["capture_date"], exif["gps_lat"], exif["gps_lon"],
        observation,
        seo.get("name"),
        seo.get("description"),
        seo.get("caption"),
        json.dumps(seo.get("keywords", [])),
        seo.get("contentLocation"),
        seo.get("slug"),
        json.dumps(schema_org, indent=2),
        total_latency,
    ])

    print(f"  Name:     {seo.get('name')}")
    print(f"  Caption:  {seo.get('caption')}")
    print(f"  Keywords: {seo.get('keywords')}")
    print(f"  Location: {seo.get('contentLocation')}")
    print(f"  Latency:  {total_latency}ms ({t1}+{t2})")


def main():
    photos = sorted(RAW_DIR.glob("*.JPG")) + sorted(RAW_DIR.glob("*.jpg"))
    total = len(photos)

    if total == 0:
        print(f"No photos found in {RAW_DIR}")
        sys.exit(1)

    print(f"Found {total} photos. Loading model...")
    model, processor = load(MODEL_PATH)
    config = load_config(MODEL_PATH)
    print("Model loaded.\n")

    con = duckdb.connect(DB_PATH)
    ensure_table(con)

    start = time.time()
    for i, photo in enumerate(photos, 1):
        process_one(model, processor, config, con, photo, i, total)

    con.close()
    elapsed = int(time.time() - start)
    print(f"\nDone. {total} photos processed in {elapsed}s ({elapsed // 60}m {elapsed % 60}s)")


if __name__ == "__main__":
    main()
