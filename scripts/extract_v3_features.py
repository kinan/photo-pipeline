
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

STYLE_PROMPT = """You are an expert in black and white photography style analysis. 
Analyze the artistic style of this image. 
Return exactly 5 keywords that describe the style (e.g., chiaroscuro, high-grain, minimalist, humanist, deep-shadows, soft-focus, etc.).
Return ONLY the keywords separated by commas."""

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
    response = requests.post(url, json=payload)
    return response.json()["embedding"]

# --- Extraction Classes ---

class StyleExtractor:
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
            # Handle potential return of ModelOutput instead of direct tensor
            if hasattr(outputs, "pooler_output"):
                image_features = outputs.pooler_output
            elif isinstance(outputs, (list, tuple)):
                image_features = outputs[0]
            else:
                image_features = outputs
        
        clip_emb = image_features.detach().cpu().numpy().flatten()
        clip_emb = clip_emb / np.linalg.norm(clip_emb)
        
        # 2. VLM Style Keywords
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": STYLE_PROMPT},
            ],
        }]
        formatted = apply_chat_template(self.vlm_processor, self.vlm_config, messages)
        # mlx_vlm.generate expects image_path or PIL Image depending on version, 
        # usually path is safer for some versions, but let's try path.
        output = mlx_generate(self.vlm_model, self.vlm_processor, formatted, str(image_path), verbose=False)
        style_keywords = output.text.strip()
        
        # 3. Nomic Embedding of Keywords
        style_emb = np.array(get_nomic_embedding(style_keywords))
        style_emb = style_emb / np.linalg.norm(style_emb)
        
        return {
            "clip_emb": clip_emb.tolist(),
            "style_keywords": style_keywords,
            "style_emb": style_emb.tolist()
        }

def init_db(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS style_v3 (
            photo_id       VARCHAR PRIMARY KEY,
            image_path     VARCHAR NOT NULL,
            clip_emb       FLOAT[768],
            style_keywords TEXT,
            style_emb      FLOAT[768],
            created_at     TIMESTAMP DEFAULT now()
        )
    """)

def run_pipeline():
    con = duckdb.connect(DB_PATH)
    init_db(con)
    
    extractor = StyleExtractor()
    
    photos = sorted(p for ext in ("*.jpg", "*.JPG") for p in RAW_DIR.glob(ext))
    print(f"Processing {len(photos)} photos...")
    
    for path in photos:
        pid = get_file_hash(path)
        
        # Check if already processed
        if con.execute("SELECT 1 FROM style_v3 WHERE photo_id = ?", [pid]).fetchone():
            print(f"  Skip {path.name}")
            continue
            
        print(f"  Processing {path.name}...")
        try:
            features = extractor.extract_features(path)
            con.execute("""
                INSERT INTO style_v3 (photo_id, image_path, clip_emb, style_keywords, style_emb)
                VALUES (?, ?, ?, ?, ?)
            """, [pid, str(path.resolve()), features["clip_emb"], features["style_keywords"], features["style_emb"]])
        except Exception as e:
            print(f"  Error processing {path.name}: {e}")

    con.close()
    print("Done.")

if __name__ == "__main__":
    run_pipeline()
