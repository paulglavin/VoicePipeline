# VoicePipeline — Speaker Management System

## What this project is
A standalone speaker management UI and backend for a home voice pipeline.
Replaces the speaker enrollment/management functionality previously handled by Lucia voice AI.
The pipeline uses: GTCRN (noise suppression) → SileroVAD (VAD) → ERes2Net (speaker ID) → Granite 4.0 1B Speech (STT).

## Current build status
- [x] Database schema (pipeline/init_db.py)
- [x] Pipeline write path (pipeline/writer.py)
- [x] Retention job (pipeline/retention.py)
- [x] FastAPI application (api/main.py)
- [x] Static frontend (ui/) — HTML/CSS/JS, no framework
- [x] Dockerfile
- [x] docker-compose.yml

## Project structure
```
VoicePipeline/
├── pipeline/
│   ├── __init__.py
│   ├── init_db.py        — schema creation, run once
│   ├── writer.py         — InteractionWriter class
│   └── retention.py      — RetentionJob class (APScheduler)
├── api/
│   ├── __init__.py
│   └── main.py           — FastAPI app (TODO)
├── ui/                   — static frontend (TODO)
├── data/                 — runtime only, gitignored
│   ├── audio/
│   └── embeddings/
├── requirements.txt
├── .gitignore
├── Dockerfile            — TODO
├── docker-compose.yml    — TODO
└── HANDOFF.md
```

## Database
SQLite at /data/speaker.db (runtime path, configurable via DB_PATH env var).
WAL mode enabled. Four tables: interactions, resolutions, enrolled_speakers, settings.
One view: interaction_current_state (joins interactions with most recent resolution).

### Key schema decisions
- interactions — immutable after insert, one row per voice event
- resolutions — append-only audit trail, multiple rows per interaction allowed
- enrolled_speakers — reference_clips is a JSON array of audio paths
- settings — key/value, seeded with defaults on init

### Confidence threshold
0.75 — interactions below this go to pending queue, above are auto-confirmed.
Currently hardcoded in pipeline/writer.py:_write_db. Should move to settings table later.

## API surface (6 endpoints)
All under /api/v1/

| Method | Path | Purpose |
|--------|------|---------|
| GET | /interactions/pending | Pending queue, includes reference_clip for low-confidence matches |
| GET | /interactions/history | Full log, filterable by speaker/date |
| GET | /audio/{id} | Serve WAV file from filesystem |
| POST | /interactions/{id}/resolve | Write resolution, update speaker profile |
| GET | /settings | Read retention settings |
| PUT | /settings | Update retention settings |

### POST /interactions/{id}/resolve body
```json
{
  "action": "confirm|reject|assign|enrol|dismiss",
  "assigned_to": "Paul"
}
```
Actions confirm and assign require assigned_to.
This endpoint is atomic — writes resolution, updates interaction status,
and side-effects enrolled_speakers in a single transaction.

## Retention policy
Three settings in settings table:
- pending_retention_days (default 7) — 0 = never purge
- resolved_retention_days (default 3) — -1 = keep indefinitely
- reference_clips_per_speaker (default 5)

Resolved interactions: audio file deleted, metadata row kept permanently.
Retention job runs every 6 hours via APScheduler inside the FastAPI process.

## Runtime configuration (environment variables)
```
DB_PATH=/data/speaker.db
AUDIO_DIR=/data/audio
EMBEDDING_DIR=/data/embeddings
HOST=0.0.0.0
PORT=8000
```

## Enrolled speakers (household)
Paul, Cassandra, Haydn.
These are seeded manually — no auto-enrolment on first run.

## Key design decisions (do not change without good reason)
- Fire and forget writes — pipeline never blocks on DB write
- asyncio.to_thread for all filesystem/DB ops — keeps event loop clean
- Embedding stored both as SQLite BLOB (fast lookup) and .npy file (backup)
- Full audit trail in resolutions — never update, only append
- Reference clips rotated by count not age — keeps highest-confidence clips
- High-confidence clips (>=0.85) only added to reference_clips rotation
- No auth in v1 — relies on network boundary (home LAN only)
- SQLite not Postgres — single household, no concurrency requirement

## FastAPI app requirements (api/main.py)
- Lifespan handler: initialise DB, start APScheduler with RetentionJob
- Mount /audio as StaticFiles pointing at AUDIO_DIR
- All endpoints under /api/v1/ prefix
- CORS enabled for localhost (UI served separately in dev)
- Use InteractionWriter from pipeline.writer
- Use RetentionJob from pipeline.retention
- Pydantic models for all request/response bodies
- Return 404 with clear message if audio file has been deleted by retention

## Frontend (ui/) — build after API
- Vanilla HTML/CSS/JS, no framework
- Single page, tabs: Pending action / Enrolled speakers / Full history / Settings
- Already fully designed as interactive mockup in the planning phase
- Pending cards: unknown (red left border) and low-confidence (amber left border)
- Low-confidence cards include compare panel with two play buttons side by side
- Resolution row: Confirm | Reject | Other match (custom dropdown button, not a select)
- Settings tab: three retention controls with contextual warnings
- Serve as static files from FastAPI using StaticFiles mount
