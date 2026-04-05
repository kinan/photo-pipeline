# scripts/analyze_comparison.py
# Reads model_inferences from photos.duckdb and prints a structured comparison
# between qwen and moondream3.
#
# Run: uv run python scripts/analyze_comparison.py

import json
import duckdb
from pathlib import Path

DB_PATH = "./outputs/photos.duckdb"


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def keyword_overlap(kw_a: str, kw_b: str) -> tuple[int, int, float]:
    """Returns (|A|, |B|, jaccard similarity)."""
    try:
        a = set(k.lower() for k in json.loads(kw_a or "[]"))
        b = set(k.lower() for k in json.loads(kw_b or "[]"))
    except Exception:
        return 0, 0, 0.0
    union = a | b
    inter = a & b
    jaccard = len(inter) / len(union) if union else 0.0
    return len(a), len(b), jaccard


def divider(char="-", width=72):
    print(char * width)


def main():
    con = duckdb.connect(DB_PATH, read_only=True)

    # Check table exists
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if "model_inferences" not in tables:
        print("No model_inferences table found. Run compare_models.py first.")
        return

    models = [r[0] for r in con.execute(
        "SELECT DISTINCT model_name FROM model_inferences ORDER BY model_name"
    ).fetchall()]

    if len(models) < 2:
        print(f"Only {len(models)} model(s) in DB. Need at least 2 to compare.")
        return

    photos = [r[0] for r in con.execute(
        "SELECT DISTINCT photo_id FROM model_inferences ORDER BY photo_id"
    ).fetchall()]

    print()
    print("MODEL COMPARISON REPORT")
    divider("=")
    print(f"Models : {' | '.join(models)}")
    print(f"Photos : {len(photos)}")
    divider("=")

    # Per-photo breakdown
    latencies = {m: [] for m in models}

    for photo_id in photos:
        rows = con.execute("""
            SELECT mi.model_name, mi.observation, mi.name, mi.description,
                   mi.caption, mi.keywords, mi.content_location, mi.slug,
                   mi.vlm_latency_ms, p.original_path
            FROM model_inferences mi
            LEFT JOIN photos p ON mi.photo_id = p.photo_id
            WHERE mi.photo_id = ?
            ORDER BY mi.model_name
        """, [photo_id]).fetchall()

        if not rows:
            continue

        # Index by model
        by_model = {r[0]: r for r in rows}
        filename = Path(rows[0][9] or photo_id).name if rows[0][9] else photo_id

        print(f"\nPhoto: {filename}  [{photo_id}]")
        divider()

        for model in models:
            if model not in by_model:
                print(f"  {model}: no result")
                continue
            _, obs, name, desc, caption, kw, loc, slug, latency, _ = by_model[model]
            latencies[model].append(latency or 0)

            print(f"  [{model}]")
            print(f"    Latency    : {latency}ms")
            print(f"    Name       : {name}")
            print(f"    Description: {desc}")
            print(f"    Caption    : {caption}")
            print(f"    Keywords   : {kw}")
            print(f"    Location   : {loc or '(none)'}")
            print(f"    Slug       : {slug}")
            print(f"    Obs words  : {word_count(obs)}")

        # Cross-model comparison for this photo
        if len(models) == 2:
            m_a, m_b = models[0], models[1]
            if m_a in by_model and m_b in by_model:
                kw_a = by_model[m_a][5]
                kw_b = by_model[m_b][5]
                n_a, n_b, jaccard = keyword_overlap(kw_a, kw_b)
                print(f"\n  Keyword overlap  : {jaccard:.0%} Jaccard  ({n_a} vs {n_b} keywords)")

                cap_a = word_count(by_model[m_a][4])
                cap_b = word_count(by_model[m_b][4])
                print(f"  Caption length   : {cap_a} words ({m_a}) vs {cap_b} words ({m_b})")

                obs_a = word_count(by_model[m_a][1])
                obs_b = word_count(by_model[m_b][1])
                print(f"  Observation len  : {obs_a} words ({m_a}) vs {obs_b} words ({m_b})")

        divider()

    # Summary statistics
    print()
    print("SUMMARY")
    divider("=")
    for model in models:
        lats = latencies[model]
        if lats:
            avg = sum(lats) / len(lats)
            print(f"  {model}")
            print(f"    Avg latency : {avg:.0f}ms  ({avg/1000:.1f}s)")
            print(f"    Min / Max   : {min(lats)}ms / {max(lats)}ms")
            print(f"    Total       : {sum(lats)}ms")

    if len(models) == 2:
        m_a, m_b = models[0], models[1]
        lats_a = latencies[m_a]
        lats_b = latencies[m_b]
        if lats_a and lats_b:
            avg_a = sum(lats_a) / len(lats_a)
            avg_b = sum(lats_b) / len(lats_b)
            faster = m_a if avg_a < avg_b else m_b
            ratio = max(avg_a, avg_b) / min(avg_a, avg_b)
            print(f"\n  {faster} is {ratio:.1f}x faster on average")

    divider("=")
    con.close()


if __name__ == "__main__":
    main()
