# SEO Pipeline Strategy: 1,000 Images on M4 Pro

## Context
Current pipeline runs 2 sequential VLM calls per image (Observer → SEO) via local MLX. At 44s/image (7B) or 29s/image (3B) this would take 8–12 hours for 1,000 images with no fine-tuning capability. The strategy is a 3-phase bootstrap → curate → fine-tune loop that cuts processing to ~5 minutes for 1,000 images and produces a domain-specific local model by the end.

---

## Architecture

```
Phase 1 — Bootstrap (Claude API)
  data/raw/*.JPG
    → file_hash + extract_exif          [reuse from process_photo.py]
    → base64 encode image
    → Claude Haiku (single-pass prompt, 15 concurrent)
    → parse_vlm_json + clean_seo        [reuse from process_photo.py]
    → model_inferences (model_name='claude-haiku-4-5')
  Time: ~5 min   Cost: ~$5–8

Phase 2 — Human Curation (Notebook)
  model_inferences WHERE model_name='claude-haiku-4-5'
    → ipywidgets approval UI in model_comparison.ipynb
    → editable fields (name, description, caption, keywords, location, slug)
    → [Approve] / [Reject] writes model_inferences.approved = TRUE/FALSE
  Goal: 500–1,000 approved rows

Phase 3 — LoRA Fine-Tune (Local 3B)
  model_inferences WHERE approved=TRUE
    → export_training_data.py → outputs/training/train.jsonl (80%) + valid.jsonl (20%)
    → finetune_3b.py via mlx_vlm.lora
    → models/qwen2.5-vl-3b-seo-lora/adapters.safetensors
  Time: ~40 min   Cost: $0
  Result: ~29s/image, domain-specific accuracy
```

---

## Phase 1 — Claude API Batch Processor

**New file:** `scripts/claude_batch_process.py`

**Dependencies to add to pyproject.toml:**
- `anthropic>=0.49.0`
- `ipywidgets>=8.1.0`

**Key design:**
- `anthropic.AsyncAnthropic()` + `asyncio.Semaphore(15)` for parallel calls
- Single-pass prompt merges Observer + SEO into one call (saves ~50% cost + latency)
- `ANTHROPIC_API_KEY` read from environment (already set)
- Reads from `data/raw/`, skips photos already in `model_inferences` for this model
- Writes to `model_inferences` with `model_name='claude-haiku-4-5'`
- Also writes EXIF + schema_org to `photos` table

**Functions to reuse from `scripts/process_photo.py`:**
- `file_hash(path)` — deduplication key
- `extract_exif(path)` — EXIF via exiftool subprocess
- `parse_vlm_json(raw, label)` — handles fenced/unfenced JSON
- `clean_seo(seo)` — strips banned keywords, deduplicates captions
- `build_schema_org(seo, exif, photo_id, path)` — assembles JSON-LD

**Single-pass prompt** (replaces OBSERVER_PROMPT + SEO_PROMPT):
```
You are a Technical SEO specialist. Before generating metadata, internally observe
the image with literal precision — note every specific clothing item, physical
descriptor, action, background element, and any visible text (street signs,
business names, landmarks) that could identify a location. Do not write the
observation out.

Using only what you can literally see, generate this schema.org/ImageObject JSON.
Apply every rule strictly:

NAME — wire-service title, 6–10 words, subject + specific clothing + action verb.
  BANNED: street, urban, city life, scene, moment, photography, monochrome,
          portrait, snapshot, candid, setting, environment

DESCRIPTION — one sentence ≤125 chars: [who + clothing] [doing what] [specific where].
  BANNED: urban, city, candid, street photography, setting

CAPTION — exactly two different factual sentences.
  Sentence 1: subject + specific clothing or held objects + action.
  Sentence 2: background — specific architecture, signage, vehicles, infrastructure.
  BANNED phrases: captures a moment, bustling, hustle, urban life, city life,
                  dynamic, candid, city street

KEYWORDS — exactly 12 unique keywords:
  3 subject tags (clothing/physical descriptors)
  3 action tags (what subjects are doing)
  3 location tags (specific place identifiers visible in image)
  3 visual tags (observable image properties)
  BANNED: photography, photo, image, street, urban, city, candid, setting, scene, environment

CONTENT LOCATION — "City, State" if any visible marker identifies it. Empty string if uncertain.

SLUG — lowercase hyphenated: subject-clothing-action-location. No genre words.

Return ONLY the JSON object. No preamble.

{"name":"...","description":"...","caption":"...","keywords":[...],"contentLocation":"...","slug":"..."}
```

**Cost estimate:**
- ~2,400 input tokens (prompt + image) + ~450 output tokens per image
- Haiku pricing: ~$0.0037/image → **~$3.70–8 for 1,000 images** (budget $10 with buffer)

**Time estimate:** ~5 minutes for 1,000 images at 15 concurrent, ~1.5s/call

---

## Phase 2 — Approval Workflow

**DB migration (one-time):**
```sql
ALTER TABLE model_inferences ADD COLUMN approved BOOLEAN DEFAULT NULL
```
Run from a new notebook cell. Safe — existing INSERT OR REPLACE statements use explicit column lists.

**Notebook additions to `model_comparison.ipynb`:**

Add two cells at the bottom:
1. **Migration cell** — runs the ALTER TABLE once (catches duplicate column error gracefully)
2. **Approval UI cell** — `ipywidgets` form per photo:
   - Shows image thumbnail above
   - Editable `Textarea` for name, description, caption, keywords (comma-separated), location, slug
   - `[Approve ✓]` and `[Reject ✗]` buttons write `approved=TRUE/FALSE` + any edits back to DB
   - Only shows `WHERE model_name='claude-haiku-4-5' AND approved IS NULL`

**Goal:** 500–1,000 approved rows. At 30–60s per image this is 4–8 hours across a few sessions.

---

## Phase 3 — LoRA Fine-Tuning

**New file:** `scripts/export_training_data.py`
- Reads `model_inferences WHERE approved=TRUE AND model_name='claude-haiku-4-5'`
- Builds JSONL with messages format for mlx_vlm SFT trainer:
  ```json
  {"messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "file:///abs/path/image.jpg"}},
      {"type": "text", "text": "<SINGLE_PASS_PROMPT>"}
    ]},
    {"role": "assistant", "content": "{\"name\":\"...\", ...approved JSON...}"}
  ]}
  ```
- Shuffles with fixed seed, writes 80% to `outputs/training/train.jsonl`, 20% to `valid.jsonl`

**New file:** `scripts/finetune_3b.py`
- Thin wrapper calling `mlx_vlm.lora.main` with these hyperparameters:

| Parameter | Value | Rationale |
|---|---|---|
| `--model-path` | `./models/qwen2.5-vl-3b-4bit` | Base model |
| `--lora-rank` | 16 | Standard for domain adaptation on 3B |
| `--lora-alpha` | 32 | 2× scaling (alpha/rank) |
| `--lora-dropout` | 0.05 | Light regularization for small dataset |
| `--learning-rate` | 2e-5 | Default from mlx_vlm, appropriate for 4-bit LoRA |
| `--batch-size` | 2 | Safe for 18GB; peak memory ~7GB |
| `--gradient-accumulation-steps` | 4 | Effective batch of 8 |
| `--iters` | 600 | ~10 epochs over 500 examples at eff. batch 8 |
| `--max-seq-length` | 1024 | Prompt ~650 + output ~350 = ~1000 tokens |
| `--grad-checkpoint` | True | Reduces activations ~60%, required on 18GB |
| `--train-on-completions` | True | Only train on assistant JSON, not prompt tokens |
| `--output-path` | `./models/qwen2.5-vl-3b-seo-lora/adapters.safetensors` | |

**Training time:** ~40 minutes on M4 Pro
**Expected gain:** Domain-specific accuracy matching or exceeding 7B on your photo vocabulary; same ~29s inference speed

---

## All File Changes

| File | Action | Phase |
|---|---|---|
| `pyproject.toml` | Add `anthropic>=0.49`, `ipywidgets>=8.1` | 1 |
| `scripts/claude_batch_process.py` | Create | 1 |
| `model_comparison.ipynb` | Add 2 cells (DB migration + approval UI) | 2 |
| `scripts/export_training_data.py` | Create | 3 |
| `scripts/finetune_3b.py` | Create | 3 |

**No existing scripts need modification.** `process_photo.py`, `batch_process.py`, `compare_models.py` all stay as-is.

---

## Verification

**Phase 1:**
- Run: `uv run python scripts/claude_batch_process.py`
- Check: `SELECT model_name, COUNT(*) FROM model_inferences GROUP BY model_name`
- Expect: 1,000 rows with `model_name='claude-haiku-4-5'`

**Phase 2:**
- Open `model_comparison.ipynb`, run migration cell, run approval cell
- Check: `SELECT COUNT(*) FROM model_inferences WHERE approved=TRUE`

**Phase 3:**
- Run: `uv run python scripts/export_training_data.py` → verify JSONL line count
- Run: `uv run python scripts/finetune_3b.py` → monitor val loss for plateau
- Test fine-tuned model: run `batch_process.py` pointed at `qwen2.5-vl-3b-seo-lora`, spot-check 5 photos
