# scripts/init_db.py
import duckdb

con = duckdb.connect("./outputs/pipeline.duckdb")

con.execute("""
    CREATE TABLE IF NOT EXISTS photos (
        photo_id         VARCHAR PRIMARY KEY,  -- sha256 of file
        original_path    VARCHAR NOT NULL,
        pipeline_state   VARCHAR DEFAULT 'ingested',

        -- SEO / EXIF
        camera_make      VARCHAR,
        camera_model     VARCHAR,
        lens_model       VARCHAR,
        iso              INTEGER,
        shutter_speed    VARCHAR,
        aperture         FLOAT,
        focal_length_mm  FLOAT,
        capture_date     TIMESTAMPTZ,
        gps_lat          FLOAT,
        gps_lon          FLOAT,

        -- Artistic context (VLM output)
        caption          TEXT,
        artistic_desc    TEXT,
        tonal_character  VARCHAR,
        scene_type       VARCHAR,
        vlm_latency_ms   INTEGER,

        ingested_at      TIMESTAMPTZ DEFAULT now()
    )
""")

con.close()
print("Schema ready.")