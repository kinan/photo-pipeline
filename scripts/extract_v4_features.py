
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import duckdb
import numpy as np
from pathlib import Path
import hashlib
import json
import time
import requests
import re
from mlx_vlm import load as mlx_load
from mlx_vlm.utils import load_config
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm import generate as mlx_generate

# --- Config ---
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
VLM_MODEL_ID  = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
DB_PATH       = "./outputs/photos.duckdb"
RAW_DIR       = Path("./data/raw")
DEVICE        = "mps" if torch.backends.mps.is_available() else "cpu"

# The "Command V6" prompt from your experiments
V4_AESTHETIC_PROMPT = """Act as a Technical SEO and Art Curator. Analyze this image and return ONLY a valid JSON object. No preamble. Focus on light-play, tonality, texture, shapes, and geometry. 

{
  "seo_metadata": {
    "suggested_filename": "keyword-rich-slug",
    "alt_text": "concise-description",
    "json_ld_keywords": ["list", "of", "10", "tags"],
    "image_schema": {
      "@context": "https://schema.org",
      "@type": "ImageObject",
      "name": "Keyword-optimized filename",
      "contentLocation": "Identify city/setting if visible, check for Chicago markers",
      "acquireLicensePage": "URL placeholder",
      "abstract": "A sophisticated 3-sentence gallery-style description"
    }
  },
  "technical_analysis": {
    "visual_geometry": "describe leading lines or symmetry",
    "lighting_style": "e.g. high-contrast, natural, chiaroscuro",
    "tonality": "describe the tonal range and texture"
  },
  "curator_commentary": {
    "style": "Leica-minimalist",
    "narrative_caption": "3-sentence moody caption",
    "mood_profile": "2-word vibe description"
  }
}

Ensure the output is valid JSON."""

# --- Helpers ---

def get_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()[:16]

def get_nomic_embedding(text: str):
    """Get embedding from local Ollama instance."""
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": "nomic-embed-text", "prompt": text}
    try:
        response = requests.post(url, json=payload)
        return response.json()["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0.0] * 768

def clean_json_output(raw_text: str) -> str:
    """Extract JSON block from model output."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        return match.group(0)
    return raw_text

# --- Extraction Classes ---

class StyleExtractorV4:
    def __init__(self):
        print(f"Loading CLIP model {CLIP_MODEL_ID} on {DEVICE}...")
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        
        print(f"Loading VLM model {VLM_MODEL_ID}...")
        self.vlm_model, self.vlm_processor = mlx_load(VLM_MODEL_ID)
        self.vlm_config = load_config(VLM_MODEL_ID)

    def extract_features(self, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        
        # 1. CLIP Visual Embedding
        inputs = self.clip_processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
            if hasattr(outputs, "pooler_output"):
                image_features = outputs.pooler_output
            elif isinstance(outputs, (list, tuple)):
                image_features = outputs[0]
            else:
                image_features = outputs
        
        clip_emb = image_features.detach().cpu().numpy().flatten()
        clip_emb = clip_emb / (np.linalg.norm(clip_emb) + 1e-8)
        
        # 2. VLM Rich Aesthetic Profile
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": V4_AESTHETIC_PROMPT},
            ],
        }]
        formatted = apply_chat_template(self.vlm_processor, self.vlm_config, messages)
        output = mlx_generate(self.vlm_model, self.vlm_processor, formatted, str(image_path), verbose=False)
        
        json_text = clean_json_output(output.text.strip())
        try:
            aesthetic_profile = json.loads(json_text)
        except Exception as e:
            print(f"  Warning: JSON parse failed for {image_path.name}: {e}")
            aesthetic_profile = {"error": "parse_failed", "raw": output.text.strip()}
        
        # 3. Create a weighted profile string for embedding
        # We combine keywords, technical analysis, and curator style for the concept vector
        profile_parts = []
        if "seo_metadata" in aesthetic_profile:
            profile_parts.append(", ".join(aesthetic_profile["seo_metadata"].get("json_ld_keywords", [])))
        if "technical_analysis" in aesthetic_profile:
            ta = aesthetic_profile["technical_analysis"]
            profile_parts.append(f"{ta.get('visual_geometry', '')} {ta.get('lighting_style', '')} {ta.get('tonality', '')}")
        if "curator_commentary" in aesthetic_profile:
            cc = aesthetic_profile["curator_commentary"]
            profile_parts.append(f"{cc.get('style', '')} {cc.get('mood_profile', '')}")
        
        profile_string = " ".join(profile_parts).strip()
        if not profile_string:
            profile_string = "black and white street photography"
            
        style_emb = np.array(get_nomic_embedding(profile_string))
        style_emb = style_emb / (np.linalg.norm(style_emb) + 1e-8)
        
        return {
            "clip_emb": clip_emb.tolist(),
            "aesthetic_json": json.dumps(aesthetic_profile),
            "style_emb": style_emb.tolist()
        }

def init_db(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS style_v4 (
            photo_id       VARCHAR PRIMARY KEY,
            image_path     VARCHAR NOT NULL,
            clip_emb       FLOAT[768],
            aesthetic_json TEXT,
            style_emb      FLOAT[768],
            created_at     TIMESTAMP DEFAULT now()
        )
    """)

def run_pipeline():
    con = duckdb.connect(DB_PATH)
    init_db(con)
    
    extractor = StyleExtractorV4()
    
    photos = sorted(p for ext in ("*.jpg", "*.JPG") for p in RAW_DIR.glob(ext))
    print(f"Processing {len(photos)} photos with V4 Pipeline...")
    
    for path in photos:
        pid = get_file_hash(path)
        
        if con.execute("SELECT 1 FROM style_v4 WHERE photo_id = ?", [pid]).fetchone():
            print(f"  Skip {path.name}")
            continue
            
        print(f"  Processing {path.name}...")
        try:
            features = extractor.extract_features(path)
            con.execute("""
                INSERT INTO style_v4 (photo_id, image_path, clip_emb, aesthetic_json, style_emb)
                VALUES (?, ?, ?, ?, ?)
            """, [pid, str(path.resolve()), features["clip_emb"], features["aesthetic_json"], features["style_emb"]])
        except Exception as e:
            print(f"  Error processing {path.name}: {e}")

    con.close()
    print("V4 Ingestion Done.")

if __name__ == "__main__":
    run_pipeline()
