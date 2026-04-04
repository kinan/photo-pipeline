# scripts/process_photo.py
import subprocess, json, hashlib, sys, time
import duckdb
from pathlib import Path
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL_PATH = "./models/qwen2.5-vl-7b-4bit"
DB_PATH    = "./outputs/photos.duckdb"

CAPTION_PROMPT = "Write one sentence describing the subject and setting of this photograph."

ARTISTIC_PROMPT = """Describe this photograph for an artist or curator. Cover:
- Light quality and direction
- Tonal range and contrast
- Composition and framing
- Mood or atmosphere
Keep it under 100 words."""

def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]

def extract_exif(path: str) -> dict:
    result = subprocess.run(
        ["exiftool", "-json", "-n", path],  # -n returns numeric values
        capture_output=True, text=True
    )
    raw = json.loads(result.stdout)[0]
    return {
        "camera_make":     raw.get("Make"),
        "camera_model":    raw.get("Model"),
        "lens_model":      raw.get("LensModel"),
        "iso":             raw.get("ISO"),
        "shutter_speed":   raw.get("ExposureTime"),
        "aperture":        raw.get("FNumber"),
        "focal_length_mm": raw.get("FocalLength"),
        "capture_date":    raw.get("DateTimeOriginal"),
        "gps_lat":         raw.get("GPSLatitude"),
        "gps_lon":         raw.get("GPSLongitude"),
    }

def run_vlm(model, processor, config, image_path: str, prompt: str) -> tuple[str, int]:
    formatted = apply_chat_template(processor, config, prompt, num_images=1)
    t0 = time.time()
    output = generate(model, processor, formatted, image_path, max_tokens=200, verbose=False)
    latency_ms = int((time.time() - t0) * 1000)
    return output.strip(), latency_ms

def process(image_path: str):
    path = str(Path(image_path).resolve())
    photo_id = file_hash(path)

    print(f"[{photo_id}] Extracting EXIF...")
    exif = extract_exif(path)

    print(f"[{photo_id}] Loading model...")
    model, processor = load(MODEL_PATH)
    config = load_config(MODEL_PATH)

    print(f"[{photo_id}] Generating caption...")
    caption, _ = run_vlm(model, processor, config, path, CAPTION_PROMPT)

    print(f"[{photo_id}] Generating artistic description...")
    artistic_desc, latency = run_vlm(model, processor, config, path, ARTISTIC_PROMPT)

    con = duckdb.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            photo_id        VARCHAR PRIMARY KEY,
            original_path   VARCHAR,
            pipeline_state  VARCHAR,
            camera_make     VARCHAR,
            camera_model    VARCHAR,
            lens_model      VARCHAR,
            iso             INTEGER,
            shutter_speed   DOUBLE,
            aperture        DOUBLE,
            focal_length_mm DOUBLE,
            capture_date    VARCHAR,
            gps_lat         DOUBLE,
            gps_lon         DOUBLE,
            caption         VARCHAR,
            artistic_desc   VARCHAR,
            vlm_latency_ms  INTEGER
        )
    """)
    con.execute("""
        INSERT OR REPLACE INTO photos (
            photo_id, original_path, pipeline_state,
            camera_make, camera_model, lens_model,
            iso, shutter_speed, aperture, focal_length_mm,
            capture_date, gps_lat, gps_lon,
            caption, artistic_desc, vlm_latency_ms
        ) VALUES (?, ?, 'inferred', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        photo_id, path,
        exif["camera_make"], exif["camera_model"], exif["lens_model"],
        exif["iso"], exif["shutter_speed"], exif["aperture"], exif["focal_length_mm"],
        exif["capture_date"], exif["gps_lat"], exif["gps_lon"],
        caption, artistic_desc, latency
    ])
    con.close()

    print(f"\nCaption:    {caption}")
    print(f"Artistic:   {artistic_desc}")
    print(f"Camera:     {exif['camera_make']} {exif['camera_model']}")
    print(f"Lens:       {exif['lens_model']}")
    print(f"Latency:    {latency}ms")

if __name__ == "__main__":
    process(sys.argv[1])