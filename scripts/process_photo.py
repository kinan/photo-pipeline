# scripts/process_photo.py

# Pipeline: EXIF → VLM (observe + SEO in one call)

import subprocess, json, hashlib, sys, time, re
import duckdb
from pathlib import Path
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL_PATH = "./models/qwen2.5-vl-7b-4bit"
DB_PATH    = "./outputs/photos.duckdb"

# --- Prompts ---

PIPELINE_PROMPT = """Look at this photograph and describe only what you literally see.

Answer each point precisely:
- Primary subject: who or what is in the foreground? What are they doing? Describe their expression, posture, clothing, and any objects they hold.
- Secondary elements: what is in the mid-ground and background? Name specific objects, people, vehicles, and architecture.
- Location markers: look carefully for any street signs, business names, transit signage, recognizable buildings, landmarks, or architectural details that could identify the city or neighborhood. Name them explicitly if visible.
- Frame: is the image black and white or color? What is the aspect ratio — portrait, landscape, or square?
- Light: what does the light tell you about time of day and weather? Where are the shadows falling?

Write a single dense paragraph. Be specific and literal. Do not interpret.

Then, using only what you described above as your source of truth, generate the following schema.org/ImageObject metadata.

BANNED words across all fields: street, urban, city, city life, scene, moment, photography, photo, image, black and white, monochrome, portrait, snapshot, candid, setting, environment, captures a moment, bustling, hustle, urban life, dynamic, city street

NAME: Wire-service title — subject with specific clothing detail, clear action verb, location or object if present. Minimum 6 words.

DESCRIPTION: One sentence under 125 characters. Format: [who with specific clothing] [doing what] [specific where]. Only use details confirmed in the observation.

CAPTION: Exactly two sentences, factual only. Sentence 1: subject, specific clothing or objects, action. Sentence 2: background — specific architecture, signage, vehicles, or infrastructure. The two sentences must be different.

KEYWORDS: Exactly 12 unique keywords — 3 subject (clothing/physical descriptors), 3 action, 3 location (specific place identifiers), 3 visual (observable image properties). Draw only from visible elements.

CONTENTLOCATION: City and state from any visible marker. Empty if genuinely uncertain.

SLUG: Lowercase hyphenated slug: subject-clothing-action-location. No genre words.

---

Output these fields immediately after your observation paragraph, one per line, in exactly this format. No preamble between them.

NAME: value
DESCRIPTION: value
CAPTION: value
KEYWORDS: keyword1, keyword2, ..., keyword12
CONTENTLOCATION: City, State or empty
SLUG: value"""


# --- Helpers ---

def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def extract_exif(path: str) -> dict:
    result = subprocess.run(
        ["exiftool", "-json", "-n", path],
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


def run_vlm(model, processor, config, image_path: str, prompt: str, max_tokens: int = 400) -> tuple[str, int]:
    formatted = apply_chat_template(processor, config, prompt, num_images=1)
    t0 = time.time()
    output = generate(model, processor, formatted, image_path, max_tokens=max_tokens, verbose=False)
    latency_ms = int((time.time() - t0) * 1000)
    return output.text.strip(), latency_ms


_KV_KEY_MAP = {"contentlocation": "contentLocation"}


def parse_kv(raw: str) -> dict:
    result = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if not key or not value:
            continue
        key = _KV_KEY_MAP.get(key, key)
        if key == "keywords":
            result[key] = [k.strip() for k in value.split(",") if k.strip()]
        else:
            result[key] = value
    return result


BANNED_KEYWORDS = {
    "photography", "photo", "image", "street", "urban", "city",
    "candid", "setting", "scene", "environment", "urban feel",
    "urban street", "urban setting", "city street", "monochrome",
}


def clean_seo(seo: dict) -> dict:
    # Strip banned words from keywords list
    seo["keywords"] = list(dict.fromkeys([
        k for k in seo.get("keywords", [])
        if k.lower() not in BANNED_KEYWORDS
    ]))

    # Deduplicate caption sentences — fix model repetition loops
    if seo.get("caption"):
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', seo["caption"]) if s.strip()]
        seen_s = set()
        unique = [s for s in sentences if s.lower() not in seen_s and not seen_s.add(s.lower())]
        seo["caption"] = " ".join(unique)

    # Warn if banned words slipped into name or caption
    banned_in_text = {"urban", "city", "candid", "street photography", "bustling", "hustle"}
    for field in ("name", "caption", "description"):
        val = seo.get(field, "") or ""
        found = [w for w in banned_in_text if w in val.lower()]
        if found:
            print(f"  [warn] banned words in {field}: {found}", file=sys.stderr)

    return seo


def build_schema_org(seo: dict, exif: dict, photo_id: str, path: str) -> dict:
    """Assemble a full schema.org/ImageObject JSON-LD block."""
    return {
        "@context": "https://schema.org",
        "@type": "ImageObject",
        "identifier": photo_id,
        "name": seo.get("name", ""),
        "description": seo.get("description", ""),
        "caption": seo.get("caption", ""),
        "keywords": ", ".join(seo.get("keywords", [])),
        "contentLocation": {
            "@type": "Place",
            "name": seo.get("contentLocation", "")
        },
        "creditText": "Kinan Sweidan",
        "creator": {
            "@type": "Person",
            "name": "Kinan Sweidan"
        },
        "encodingFormat": "image/jpeg",
        "acquireLicensePage": "",
        "dateCreated": exif.get("capture_date", ""),
        "exifData": [
            {"@type": "PropertyValue", "name": "Make",          "value": exif.get("camera_make", "")},
            {"@type": "PropertyValue", "name": "Model",         "value": exif.get("camera_model", "")},
            {"@type": "PropertyValue", "name": "LensModel",     "value": exif.get("lens_model", "")},
            {"@type": "PropertyValue", "name": "ISO",           "value": exif.get("iso", "")},
            {"@type": "PropertyValue", "name": "ExposureTime",  "value": exif.get("shutter_speed", "")},
            {"@type": "PropertyValue", "name": "FNumber",       "value": exif.get("aperture", "")},
            {"@type": "PropertyValue", "name": "FocalLength",   "value": exif.get("focal_length_mm", "")},
        ],
    }


# --- Main pipeline ---

def process(image_path: str, model, processor, config):
    path = str(Path(image_path).resolve())
    photo_id = file_hash(path)

    print(f"[{photo_id}] Extracting EXIF...")
    exif = extract_exif(path)

    print(f"[{photo_id}] Generating metadata...")
    raw_output, t1 = run_vlm(model, processor, config, path, PIPELINE_PROMPT, max_tokens=200)

    kv_start = re.search(r"(?m)^NAME:", raw_output, re.IGNORECASE)
    observation = raw_output[:kv_start.start()].strip() if kv_start else ""
    seo_raw = raw_output[kv_start.start():] if kv_start else raw_output

    seo = clean_seo(parse_kv(seo_raw))

    total_latency = t1
    schema_org = build_schema_org(seo, exif, photo_id, path)

    con = duckdb.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            photo_id            VARCHAR PRIMARY KEY,
            original_path       VARCHAR,
            pipeline_state      VARCHAR,
            -- EXIF
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
            -- Observer
            observation         VARCHAR,
            -- SEO (schema.org ImageObject fields)
            name                VARCHAR,
            description         VARCHAR,
            caption             VARCHAR,
            keywords            VARCHAR,
            content_location    VARCHAR,
            slug                VARCHAR,
            -- Full schema.org JSON-LD blob
            schema_org          VARCHAR,
            -- Meta
            vlm_latency_ms      INTEGER
        )
    """)
    con.execute("""
        INSERT OR REPLACE INTO photos (
            photo_id, original_path, pipeline_state,
            camera_make, camera_model, lens_model,
            iso, shutter_speed, aperture, focal_length_mm,
            capture_date, gps_lat, gps_lon,
            observation,
            name, description, caption, keywords, content_location, slug,
            schema_org,
            vlm_latency_ms
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
    con.close()

    print(f"\nName:        {seo.get('name')}")
    print(f"Slug:        {seo.get('slug')}")
    print(f"Description: {seo.get('description')}")
    print(f"Caption:     {seo.get('caption')}")
    print(f"Keywords:    {seo.get('keywords')}")
    print(f"Location:    {seo.get('contentLocation')}")
    print(f"Camera:      {exif['camera_make']} {exif['camera_model']}")
    print(f"Latency:     {total_latency}ms")


if __name__ == "__main__":
    print("Loading model...")
    model, processor = load(MODEL_PATH)
    config = load_config(MODEL_PATH)
    for image_path in sys.argv[1:]:
        process(image_path, model, processor, config)
