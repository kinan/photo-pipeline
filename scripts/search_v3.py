
import torch
from transformers import CLIPProcessor, CLIPModel
import duckdb
import numpy as np
from pathlib import Path
from PIL import Image
import sys
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
    response = requests.post(url, json=payload)
    return response.json()["embedding"]

class StyleSearcher:
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
        return emb / np.linalg.norm(emb)

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
        return emb / np.linalg.norm(emb)

    def search(self, query_type: str, query_val: str, top_n: int = 5):
        if query_type == "text":
            q_emb = self.get_text_embedding(query_val)
        else:
            q_emb = self.get_image_embedding(Path(query_val))

        rows = self.con.execute("SELECT image_path, clip_emb, style_keywords FROM style_v3").fetchall()
        
        scored = []
        for path, clip_emb, keywords in rows:
            sim = np.dot(q_emb, np.array(clip_emb))
            scored.append((sim, path, keywords))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_n]

    def cluster_styles(self):
        import umap
        rows = self.con.execute("SELECT image_path, clip_emb, style_emb FROM style_v3").fetchall()
        if not rows:
            return None
        
        paths = [r[0] for r in rows]
        # Hybrid embedding: 50% CLIP Visual, 50% Style Conceptual
        # Keywords often contain more 'distinguishable' style markers for B&W
        clip_matrix = np.array([r[1] for r in rows])
        style_matrix = np.array([r[2] for r in rows])
        
        combined = np.hstack([clip_matrix * 0.5, style_matrix * 0.5])

        # Use UMAP to reduce to 15 dimensions to help HDBSCAN find tighter clusters
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=15, random_state=42)
        reduced_data = reducer.fit_transform(combined)
        
        clusterer = HDBSCAN(min_cluster_size=5, min_samples=2, metric='euclidean')
        labels = clusterer.fit_predict(reduced_data)
        
        clusters = {}
        for label, path in zip(labels, paths):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(Path(path).name)
        
        return clusters

if __name__ == "__main__":
    searcher = StyleSearcher()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/search_v3.py search-text 'grainy high contrast street photo'")
        print("  python scripts/search_v3.py search-image data/raw/Kinan.Sweidan-1.jpg")
        print("  python scripts/search_v3.py cluster")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "search-text":
        query = sys.argv[2]
        results = searcher.search("text", query)
        print(f"\nResults for text query: '{query}'")
        for sim, path, keywords in results:
            print(f"  [{sim:.3f}] {Path(path).name} | Keywords: {keywords}")
            
    elif cmd == "search-image":
        query = sys.argv[2]
        results = searcher.search("image", query)
        print(f"\nResults for image query: '{Path(query).name}'")
        for sim, path, keywords in results:
            print(f"  [{sim:.3f}] {Path(path).name} | Keywords: {keywords}")
            
    elif cmd == "cluster":
        clusters = searcher.cluster_styles()
        print("\nStyle Clusters (HDBSCAN):")
        for label, photos in clusters.items():
            cluster_name = "Outliers" if label == -1 else f"Cluster {label}"
            print(f"  {cluster_name}: {len(photos)} photos")
            print(f"    {', '.join(photos[:5])}...")
