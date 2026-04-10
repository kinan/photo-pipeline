## Goal
This project aims to enable contextual understanding of photographs to increase the user's ability to discover and group photos.

## High-Level Business Requirements
Build a solution that enables the following requirements:
1) Extract context from photographs to improve SEO and website search discoverability. 
2) Provide artistic descriptions to be used for website content and discoverability. 
3) Group photos by style, artistic elements, and patterns. 

## Technical Level Technical Requirements
Create a batch data pipeline that achieves: 
1) The architecture should follow data engineering best practices. When short-term solutions are presented, they should be weighed against long-term solutions
2) The solutions should be processed for the first 2,000 photos on the following hardware 
	1) Dedicated Mac Mini M4 with 16 GB memory (server)
	2) Development/ Analysis M4 Pro with 24 GB memory (server)
3) We want to be able to scale to a cloud architecture when we grow
4) We need a database that supports data analysis and LLM fine-tuning
5) We want to consider, but necessarily choose the following solutions: dockDB, Qwen2.5-VL, CLIP (ViT-B/32), MLX framework, Moondream3, Faiss


## Candidate Technology Summary

| Technology        | Layer | Role                                                         |
| ----------------- | ----- | ------------------------------------------------------------ |
| **MLX**           | 2     | Execution framework for all inference on Apple Silicon       |
| **Qwen2.5-VL**    | 2     | Primary VLM for captions and artistic description            |
| **CLIP ViT-B/32** | 3     | Visual embedding generation                                  |
| **Faiss**         | 3     | Exact similarity index and nearest-neighbor retrieval        |
| **DuckDB**        | 4     | Structured output storage, analytics, and fine-tuning export |

Prompt

**Scope the architecture to these four layers — treat each as a separate section:**

1. `**Ingestion & Storage** — How photos enter the pipeline and where raw assets and metadata are persisted`
2. `**ML Inference** — Image understanding: caption generation, artistic description, and metadata extraction`
3. `**Vector Similarity & Grouping** — Embedding generation and clustering by style, artistic elements, and patterns`
4. `**Database** — Storage for structured outputs supporting both data analysis queries and LLM fine-tuning dataset export`


## Questions To Be Answered
What is the caption of the photo?
How is it categorized?
What are other relevant photos or similar in style? 
What photos might this customer like based on their last purchase? I want to predict which photo a customer might be interested in based on their behavior on the website. 

**Status:** Planning / Build phase  
**Created:** 2025-04

---

## The Core Idea

```
Image → Claude Vision (expensive, accurate) → Ground truth JSON
Image → Qwen local (free, fast)             → Compare + Score
                                                      ↓
                                             Fine-tune when quality drops
```

Claude is the teacher. Qwen is the student. You pay for Claude only once per image (backfill), and Qwen handles new uploads for free.

---

## Stack

|Component|Tool|
|---|---|
|Language|Python 3.11+|
|Package manager|uv|
|Vision (cloud)|Claude claude-opus-4-5 via Anthropic SDK|
|Vision (local)|`qwen2.5vl:7b` via Ollama|
|Embeddings (local)|`nomic-embed-text` via Ollama|
|Database|DuckDB (`vision_seo.duckdb`)|
|Analysis|JupyterLab + pandas + matplotlib|
|CLI|Click or argparse|

---

## Ollama Models

```bash
ollama pull qwen2.5vl:7b        # vision analysis
ollama pull nomic-embed-text    # semantic similarity scoring
```

Why `qwen2.5vl:7b` over `llama3.2-vision:11b`:

- Outperforms Llama 3.2 at smaller footprint
- Better at structured JSON output
- Uses ~6GB of unified memory — leaves headroom on 24GB M4


---

## Build Phases

- [[Prompts Experemntations]]
- [[seo-pipeline-strategy]]

---


