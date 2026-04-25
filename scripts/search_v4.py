
import torch
from transformers import CLIPProcessor, CLIPModel
import duckdb
import numpy as np
from pathlib import Path
from PIL import Image
import sys
import json
import requests
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

# --- Config ---
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
DB_PATH       = "./outputs/photos.duckdb"
DEVICE        = "mps" if torch.backends.mps.is_available() else "cpu"

def get_nomic_embedding(text: str):
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": "nomic-embed-text", "prompt": text}
    try:
        response = requests.post(url, json=payload)
        return response.json()["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0.0] * 768

class StyleSearcherV4:
    def __init__(self):
        print(f"Loading CLIP model {CLIP_MODEL_ID} on {DEVICE}...")
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        self.con = duckdb.connect(DB_PATH)

    def get_text_embedding(self, text: str):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            if hasattr(outputs, "pooler_output"):
                text_features = outputs.pooler_output
            elif isinstance(outputs, (list, tuple)):
                text_features = outputs[0]
            else:
                text_features = outputs
        emb = text_features.detach().cpu().numpy().flatten()
        return emb / (np.linalg.norm(emb) + 1e-8)

    def get_image_embedding(self, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            if hasattr(outputs, "pooler_output"):
                image_features = outputs.pooler_output
            elif isinstance(outputs, (list, tuple)):
                image_features = outputs[0]
            else:
                image_features = outputs
        emb = image_features.detach().cpu().numpy().flatten()
        return emb / (np.linalg.norm(emb) + 1e-8)

    def search(self, query_type: str, query_val: str, top_n: int = 5):
        if query_type == "text":
            # For text queries, we compare against BOTH visual and conceptual space
            # but conceptual (style_emb) is usually more relevant for style keywords
            q_clip_emb = self.get_text_embedding(query_val)
            q_style_emb = np.array(get_nomic_embedding(query_val))
            q_style_emb = q_style_emb / (np.linalg.norm(q_style_emb) + 1e-8)
        else:
            q_clip_emb = self.get_image_embedding(Path(query_val))
            q_style_emb = None # Not used for image queries in basic mode

        rows = self.con.execute("SELECT image_path, clip_emb, style_emb, aesthetic_json FROM style_v4").fetchall()
        
        scored = []
        for path, clip_emb, style_emb, aesthetic_json in rows:
            if query_type == "text":
                # Weighted hybrid search for text
                sim_clip = np.dot(q_clip_emb, np.array(clip_emb))
                sim_style = np.dot(q_style_emb, np.array(style_emb))
                sim = (sim_clip * 0.4) + (sim_style * 0.6)
            else:
                # Direct visual search for images
                sim = np.dot(q_clip_emb, np.array(clip_emb))
            
            try:
                meta = json.loads(aesthetic_json)
                vibe = meta.get("curator_commentary", {}).get("mood_profile", "unknown")
            except:
                vibe = "unknown"
                
            scored.append((sim, path, vibe))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_n]

    def cluster_styles(self):
        import umap
        rows = self.con.execute("SELECT image_path, clip_emb, style_emb FROM style_v4").fetchall()
        if not rows:
            return None
        
        paths = [r[0] for r in rows]
        # Hybrid embedding: 40% CLIP Visual, 60% Qwen Conceptual Aesthetic
        clip_matrix = np.array([r[1] for r in rows])
        style_matrix = np.array([r[2] for r in rows])
        
        combined = np.hstack([clip_matrix * 0.4, style_matrix * 0.6])

        # UMAP for high-quality manifold projection
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.05, n_components=10, random_state=42)
        reduced_data = reducer.fit_transform(combined)
        
        # HDBSCAN for natural cluster discovery
        clusterer = HDBSCAN(min_cluster_size=4, min_samples=2, metric='euclidean')
        labels = clusterer.fit_predict(reduced_data)
        
        clusters = {}
        for label, path in zip(labels, paths):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(Path(path).name)
        
        return clusters

if __name__ == "__main__":
    searcher = StyleSearcherV4()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/search_v4.py search-text 'chiaroscuro cinematic street portrait'")
        print("  python scripts/search_v4.py search-image data/raw/Kinan.Sweidan-1.jpg")
        print("  python scripts/search_v4.py cluster")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "search-text":
        query = sys.argv[2]
        results = searcher.search("text", query)
        print(f"\nV4 Results for text query: '{query}'")
        for sim, path, vibe in results:
            print(f"  [{sim:.3f}] {Path(path).name} | Vibe: {vibe}")
            
    elif cmd == "search-image":
        query = sys.argv[2]
        results = searcher.search("image", query)
        print(f"\nV4 Results for image query: '{Path(query).name}'")
        for sim, path, vibe in results:
            print(f"  [{sim:.3f}] {Path(path).name} | Vibe: {vibe}")
            
    elif cmd == "cluster":
        clusters = searcher.cluster_styles()
        print("\nV4 Style Clusters (Hybrid UMAP + HDBSCAN):")
        for label, photos in sorted(clusters.items()):
            cluster_name = "Outliers" if label == -1 else f"Style Group {label}"
            print(f"  {cluster_name}: {len(photos)} photos")
            print(f"    {', '.join(photos[:5])}...")
