# VoicePipeline — Speaker Management System
## Project handoff document

---

## What we're building and why

This system replaces the speaker enrollment and management UI from Lucia voice AI.
Lucia had a good interface for reviewing voice detections and promoting/merging speaker
profiles, but it didn't show what was actually said. This version adds the transcript
to every interaction so you can see "Unknown speaker said 'turn the kitchen lights off'
at 14:32" before deciding whether to enrol or merge.

The broader pipeline this feeds into uses four models in sequence:
1. **GTCRN Full (Streaming)** — noise suppression, runs as a PipeWire LADSPA filter or ONNX service
2. **SileroVAD v5** — voice activity detection, gates everything downstream
3. **3D-Speaker ERes2Net** — speaker identification, produces a 192-dim embedding vector
4. **Granite 4.0 1B Speech** — STT, produces the transcript

This system is the management layer on top of that pipeline — it stores every
interaction, surfaces the ones that need human review, and lets you resolve them.

---

## Architecture decisions

### Why SQLite and not Postgres
Single household, three enrolled speakers. No concurrency requirement that SQLite
can't handle with WAL mode enabled. One less service to run.

### Why fire and forget writes
The voice pipeline should return a transcript to the voice system as fast as possible.
A database write shouldn't add latency to that path. If a write fails, it's logged
and the interaction is lost — acceptable for a home system.

### Why an audit trail in resolutions
The resolutions table is append-only. Every action taken in the UI (confirm, reject,
assign, re-assign) writes a new row. The current state is always the most recent row.
This lets you answer questions like "how often did the model suggest Haydn but we
assigned to Paul?" — useful for tuning confidence thresholds over time.

### Why embeddings stored in two places
In the SQLite BLOB for fast runtime lookup (similarity comparison without hitting
the filesystem). On disk as .npy as a backup in case the database is rebuilt.

### Confidence threshold (0.75)
Interactions below 0.75 match confidence go to the pending queue for human review.
Above 0.75 are auto-confirmed. This threshold is currently hardcoded in
pipeline/writer.py. Once you have a few weeks of data you'll have a feel for
whether this is too aggressive or too conservative — move it to the settings
table at that point.

### Reference clips
High-confidence clips (>=0.85) are added to each speaker's reference_clips pool.
The pool is capped at N clips (default 5, configurable in settings). These are the
"known good" clips shown in the compare panel when reviewing a low-confidence match.
Older clips rotate out as newer ones arrive.

---

## UI design

The UI was designed iteratively as part of the planning process. Key design decisions:

- **Pending tab**: two card styles — red left border for no-match unknowns, amber for
  low-confidence matches
- **Low-confidence cards**: always show the compare panel expanded (no click needed),
  with two side-by-side play buttons — the unknown clip and a known-good reference clip
- **Resolution row**: Confirm | Reject | Other match — all three on one line.
  "Other match" is a custom dropdown button (not a native select) that opens a menu
  below it. Selecting an option resolves the card immediately.
- **Cards dim and lock** once resolved — they stay visible so you can review what
  you've actioned, they just can't be changed again
- **Settings tab**: three retention controls (pending age, resolved audio age,
  reference clips per speaker) with contextual warning messages for risky choices
- **History tab**: filterable by speaker and date range

The UI mockup was built as an interactive HTML widget during planning and can be
used as a direct reference for the frontend build.

---

## What's built

### pipeline/init_db.py
Creates the database schema. Run once before first start:
```
python -m pipeline.init_db --db /data/speaker.db
```
Safe to re-run — uses CREATE TABLE IF NOT EXISTS throughout.

Tables: interactions, resolutions, enrolled_speakers, settings
Indexes: status+date, speaker+date, resolution interaction_id
View: interaction_current_state (interactions left-joined with most recent resolution)

### pipeline/writer.py
The InteractionWriter class. Instantiate once, call write() per interaction.
```python
writer = InteractionWriter(
    db_path="/data/speaker.db",
    audio_dir="/data/audio",
    embedding_dir="/data/embeddings"  # optional, defaults to audio_dir/embeddings
)

# In your orchestrator, at the end of the pipeline:
asyncio.create_task(
    writer.write(
        audio_bytes=wav_bytes,
        transcript="Turn the kitchen lights off",
        duration_ms=2400,
        vad_confidence=0.97,
        matched_speaker="Haydn",   # None if no match
        match_confidence=0.62,     # None if no match
        embedding=numpy_array,     # 192-dim float32, None if unavailable
    )
)
```

### pipeline/retention.py
The RetentionJob class. Designed for APScheduler inside the FastAPI process.
Purges pending clips past their age limit and deletes audio files from resolved
interactions. Always keeps metadata rows and embedding files.

---

## What's next (in order)

1. **api/main.py** — FastAPI application
   - Lifespan handler: init DB, start APScheduler
   - Mount /audio as StaticFiles
   - All endpoints under /api/v1/
   - CORS for localhost
   - Pydantic models for request/response

2. **ui/** — Static frontend
   - Vanilla HTML/CSS/JS, no framework
   - Based directly on the interactive mockup from planning
   - Served as StaticFiles from FastAPI

3. **Dockerfile** — Single container
   - Python 3.12 slim base
   - Copies pipeline/, api/, ui/
   - Runs init_db then uvicorn

4. **docker-compose.yml**
   - Mounts data/ as a volume
   - Exposes port 8000
   - Environment variables for paths and config

---

## Environment variables
```
DB_PATH=/data/speaker.db
AUDIO_DIR=/data/audio
EMBEDDING_DIR=/data/embeddings
HOST=0.0.0.0
PORT=8000
```

---

## Enrolled speakers
Paul, Cassandra, Haydn. Seeded manually — no auto-enrolment on first run.
A seed script or admin endpoint should be added to create the initial profiles.

---

## What this system does NOT do
- It does not run the voice pipeline itself (GTCRN, VAD, ERes2Net, Granite)
- It does not handle TTS or Home Assistant integration
- It has no authentication in v1 — home LAN only
- It does not provide a mobile app — browser-based UI only
