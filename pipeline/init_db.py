"""
init_db.py — Create the database schema.

Run once before starting the application:

    python -m pipeline.init_db --db /data/speaker.db

Safe to re-run — uses CREATE TABLE IF NOT EXISTS throughout.
"""

import argparse
import sqlite3
from pathlib import Path


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ---------------------------------------------------------------
-- Core tables
-- ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS interactions (
    id               TEXT PRIMARY KEY,
    recorded_at      TEXT NOT NULL,
    audio_path       TEXT,
    transcript       TEXT NOT NULL,
    duration_ms      INTEGER NOT NULL,
    vad_confidence   REAL NOT NULL,
    matched_speaker  TEXT,
    match_confidence REAL,
    embedding_blob   BLOB,
    embedding_path   TEXT,
    status           TEXT NOT NULL
        CHECK(status IN ('pending','resolved','dismissed')),
    created_at       TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    timings          TEXT
);



CREATE TABLE IF NOT EXISTS resolutions (
    id              TEXT PRIMARY KEY,
    interaction_id  TEXT NOT NULL
        REFERENCES interactions(id),
    resolved_at     TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    action          TEXT NOT NULL
        CHECK(action IN ('confirm','reject','assign','enrol','dismiss')),
    assigned_to     TEXT,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS enrolled_speakers (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL UNIQUE,
    created_at        TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    last_active_at    TEXT,
    interaction_count INTEGER NOT NULL DEFAULT 0,
    avg_confidence    REAL,
    reference_clips   TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ---------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_interactions_status
    ON interactions(status, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_interactions_speaker
    ON interactions(matched_speaker, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_resolutions_interaction
    ON resolutions(interaction_id, resolved_at DESC);

-- ---------------------------------------------------------------
-- View — current resolved state per interaction
-- ---------------------------------------------------------------

CREATE VIEW IF NOT EXISTS interaction_current_state AS
SELECT
    i.*,
    r.action      AS current_action,
    r.assigned_to AS current_assigned_to,
    r.resolved_at AS last_resolved_at
FROM interactions i
LEFT JOIN resolutions r ON r.id = (
    SELECT id FROM resolutions
    WHERE interaction_id = i.id
    ORDER BY resolved_at DESC
    LIMIT 1
);

-- ---------------------------------------------------------------
-- Default settings (INSERT OR IGNORE — won't overwrite existing)
-- ---------------------------------------------------------------

INSERT OR IGNORE INTO settings VALUES ('pending_retention_days',       '7');
INSERT OR IGNORE INTO settings VALUES ('resolved_retention_days',      '3');
INSERT OR IGNORE INTO settings VALUES ('reference_clips_per_speaker',  '5');
"""


def init(db_path: str) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        # Migrate existing databases: add columns introduced after initial release
        for migration in (
            "ALTER TABLE interactions ADD COLUMN timings TEXT",
        ):
            try:
                conn.execute(migration)
                conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists
        conn.commit()
        print(f"Database initialised at {db_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialise the speaker management database")
    parser.add_argument("--db", required=True, help="Path to SQLite database file")
    args = parser.parse_args()
    init(args.db)
