# VoicePipeline — Architecture

Internal reference for contributors. For installation and usage see the [README](../README.md).

---

## Overview

VoicePipeline is a FastAPI application that runs two servers:

- **Wyoming TCP server** (`:10300`) — receives audio from Home Assistant, runs the four-stage pipeline, returns a transcript
- **HTTP management API + UI** (`:8000`) — speaker enrolment, review of pending interactions, settings, model reload

Everything is single-process. The FastAPI event loop owns the Wyoming server task, the APScheduler retention jobs, and all async DB writes. Synchronous ML inference runs in the default `asyncio.to_thread` thread pool to keep the event loop unblocked.

---

## Four-stage pipeline

```
Wyoming AudioChunk(s)
        ↓
[1] NoiseSuppressor      DTLN (2 × ONNX models)
        ↓
[2] VoiceActivityDetector    SileroVAD (ONNX)
        ↓ (silent frames discarded here)
[3] SpeakerIdentifier    WeSpeaker ECAPA-TDNN (ONNX via wespeakerruntime)
        ↓
[4] SpeechToText         Local: HuggingFace transformers (Granite 4.0 1B Speech)
                         Remote: OpenAI-compatible /v1/audio/transcriptions
        ↓
InteractionWriter        SQLite + WAV + .npy (fire-and-forget asyncio.create_task)
```

Each stage degrades gracefully rather than crashing the pipeline:

| Stage failure | Behaviour |
|---|---|
| Noise suppressor | Raw audio passed downstream |
| VAD error | Full audio passed downstream (confidence set to 0.5) |
| VAD returns silence | Utterance discarded — `None` returned |
| Speaker ID error | `matched_speaker = None`, pipeline continues |
| STT returns empty | Utterance discarded — `None` returned |

Stage timings (`ns_ms`, `vad_ms`, `sid_ms`, `stt_ms`, `total_ms`) are recorded per-interaction in the `timings` JSON column and exposed in the History tab.

---

## Stage 1 — Noise suppression

**Model:** DTLN (Dual-signal Transformation LSTM Network), two ONNX models run in sequence.

**Implementation:** `pipeline/models/noise_suppressor.py`

Wet/dry blend is controlled by the `DTLN_MIX` environment variable (`0.0` = bypass, `1.0` = full suppression). Reducing it to `0.5` can help if the suppressor introduces artefacts on already-clean audio.

---

## Stage 2 — Voice activity detection

**Model:** SileroVAD (ONNX). The ONNX file ships inside the `silero_vad` pip package — no download required.

**Implementation:** `pipeline/models/vad.py`

Returns `(speech_pcm_bytes, confidence)`. If no speech is detected it returns `(None, 0.0)` and the orchestrator discards the utterance immediately, avoiding unnecessary work in stages 3 and 4.

---

## Stage 3 — Speaker identification

**Model:** WeSpeaker VoxCeleb ResNet34 (ONNX). Downloaded by `wespeakerruntime` on first use and cached in `MODEL_DIR/wespeaker/`.

**Implementation:** `pipeline/models/speaker_id.py` — `SpeakerIdentifier`

### Embedding extraction

`wespeakerruntime` takes a file path, not raw audio bytes. For live inference, the PCM is written to a `tempfile.mkstemp` WAV, passed to `Speaker.extract_embedding()`, then the temp file is deleted. The resulting embedding vector is L2-normalised before comparison.

### torchaudio monkey-patch

`torchaudio` 2.7 routes audio loading through `torchcodec` by default, which may not be present in all CUDA builds. `wespeakerruntime` calls `torchaudio.load()` internally. To guarantee a working backend, `_patch_torchaudio_load()` replaces `torchaudio.load` with a `soundfile`-based implementation at module import time.

### Speaker cache

Enrolled speakers are matched by cosine similarity against a mean embedding computed from their reference clips. The cache is held in memory as `dict[name, l2_normed_ndarray]` and rebuilt from SQLite on a 5-minute TTL (also rebuilt immediately via `force_refresh()` after any enrolment or deletion).

### Matching thresholds

Both thresholds are read from the `settings` table on every call (not cached) so management-UI changes take effect without restart:

| Setting | Default | Effect |
|---|---|---|
| `match_threshold` | 0.50 | Below this → speaker unknown |
| `confirm_threshold` | 0.75 | Above this → auto-resolve without review |

Auto-resolution uses a separate threshold in `writer.py`: clips are only added to a speaker's reference set if similarity ≥ 0.85, regardless of the `confirm_threshold`. Manual confirms and assigns always add the clip.

### Model hot-swap

Calling `POST /models/reload` constructs a new `SpeakerIdentifier` (and `SpeechToText`) in a background task, then calls `orchestrator.swap_sid(new_sid)`. The old instance is discarded. The ONNX model filename is configurable in Settings → Speaker Identification.

---

## Stage 4 — Speech-to-text

**Implementation:** `pipeline/models/stt.py` — `SpeechToText` factory

`SpeechToText.__init__` selects one of two implementations based on `stt_provider`:

### Local: `_LocalSTT`

Loads the configured HuggingFace model via `transformers.AutoModelForSpeechSeq2Seq` with `bfloat16` dtype and `device_map` auto-selecting CUDA if available.

Default model: `ibm-granite/granite-4.0-1b-speech`. The transcription prompt is applied as a chat template:

```
[{"role": "user", "content": "<|audio|>can you transcribe the speech into a written format?"}]
```

Model weights are cached in `MODEL_DIR/granite_stt/`.

### Remote: `_RemoteSTT`

Sends audio to any OpenAI-compatible `/v1/audio/transcriptions` endpoint (e.g. Ollama on a separate machine). The PCM bytes are encoded as an in-memory WAV using the `wave` module before upload — no temp file required.

Granite-specific `<|...|>` prompt tokens are stripped automatically when switching to a remote provider, since other models don't use that format.

Uses `httpx` with a 30-second timeout. The `api_key` field is optional (omitted from the request if blank).

---

## Interaction write path

**Implementation:** `pipeline/writer.py` — `InteractionWriter`

After stage 4 the orchestrator calls `asyncio.create_task(writer.write(...))` — fire-and-forget. The write never blocks the Wyoming response.

`write()` does three things, each run in `asyncio.to_thread`:

1. **WAV file** — 16-bit, 16 kHz, mono. Written to `AUDIO_DIR/{interaction_id}.wav`.
2. **Embedding file** — `numpy.save` to `EMBEDDING_DIR/{interaction_id}.npy`. Also stored as a raw `float32` blob in the `embedding_blob` column for fast in-process access.
3. **SQLite row** — single `INSERT INTO interactions`.

If the match confidence exceeds `confirm_threshold`, `write()` also:

- Sets `status = 'resolved'` immediately
- Writes a `resolutions` row with `action = 'confirm'`
- Updates the enrolled speaker's `avg_confidence` and `interaction_count`
- Appends the clip to `reference_clips` if similarity ≥ 0.85, rotating out the oldest if over the per-speaker limit

Any exception in `write()` is caught and logged — the pipeline is never blocked or crashed by a write failure.

---

## Wyoming server

**Implementation:** `pipeline/wyoming_server.py` — `WyomingServer` / `_PipelineEventHandler`

One `_PipelineEventHandler` per TCP connection. The flow per utterance:

```
HA → Describe       → Info (service description)
HA → AudioStart     → capture sample rate / format
HA → AudioChunk(s)  → accumulate bytes
HA → AudioStop      → process + respond
                    → WebhookNotifier.notify()  (before Transcript)
                    → Transcript(text=...)
                    ← close connection
```

HA reconnects for the next utterance. Multi-channel or non-16 kHz audio is down-mixed and resampled by `_normalise_audio()` before being passed to the orchestrator.

The clean transcript (no speaker prefix) is always what Wyoming returns to HA. Speaker identity is delivered out-of-band via the webhook, described below.

---

## Webhook — Personality LLM integration

**Implementation:** `pipeline/webhook.py` — `WebhookNotifier`

When a speaker is identified, `WebhookNotifier.notify()` is awaited **before** the Wyoming `Transcript` event is written. This races the webhook ahead of the transcript so [Personality LLM](https://github.com/Paul-Glavin/personality_llm) can cache the speaker identity before the conversation agent is invoked.

```
POST {ha_base_url}/api/webhook/personality_llm_input
{
  "speaker_id": "paul",
  "confidence": 0.9142,
  "timestamp": "2026-05-04T09:00:00.000000+00:00",
  "interaction_id": "<uuid>"
}
```

Personality LLM caches this for 2 seconds. When the Wyoming transcript arrives moments later, the integration reads the cached speaker name and routes the conversation to that person's profile.

The POST has a 1-second timeout. On timeout or any HTTP error the pipeline logs a warning and continues — the transcript is sent regardless.

`ha_base_url` and `ha_webhook_enabled` are re-read from the DB settings table on every call, so changes in the UI take effect without restart. Environment variables (`HA_BASE_URL`, `HA_WEBHOOK_ENABLED`) act as fallbacks if the DB values are absent.

---

## Database schema

SQLite with WAL journal mode. One file at `DB_PATH` (default `/data/speaker.db`).

### `interactions`

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | UUID |
| `recorded_at` | TEXT | ISO-8601 UTC |
| `audio_path` | TEXT | Path to WAV; nulled out by retention |
| `transcript` | TEXT | Raw STT output |
| `duration_ms` | INTEGER | Derived from VAD-trimmed PCM length |
| `vad_confidence` | REAL | 0.0–1.0 |
| `matched_speaker` | TEXT | Name or NULL |
| `match_confidence` | REAL | Cosine similarity or NULL |
| `embedding_blob` | BLOB | float32 little-endian bytes |
| `embedding_path` | TEXT | Path to `.npy` file |
| `status` | TEXT | `pending` / `resolved` / `dismissed` |
| `timings` | TEXT | JSON: `{ns, vad, sid, stt, total}` in ms |

### `resolutions`

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | UUID |
| `interaction_id` | TEXT FK | References `interactions.id` |
| `resolved_at` | TEXT | ISO-8601 UTC |
| `action` | TEXT | `confirm` / `reject` / `assign` / `enrol` / `dismiss` |
| `assigned_to` | TEXT | Speaker name for confirm/assign/enrol |

Multiple resolutions per interaction are valid (re-assign after initial confirm). The API always reads the most recent resolution via a correlated subquery.

### `enrolled_speakers`

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | UUID |
| `name` | TEXT UNIQUE | Case-sensitive; matched by Wyoming lowercase |
| `interaction_count` | INTEGER | Cumulative |
| `avg_confidence` | REAL | Rolling mean updated on each auto-confirm |
| `reference_clips` | TEXT | JSON array of WAV paths; bounded by `reference_clips_per_speaker` |

### `settings`

Key-value table. All defaults are seeded by `INSERT OR IGNORE` in `init_db.py`, which is safe to re-run on an existing database.

| Key | Default |
|---|---|
| `pending_retention_days` | `7` |
| `resolved_retention_days` | `3` |
| `reference_clips_per_speaker` | `5` |
| `match_threshold` | `0.50` |
| `confirm_threshold` | `0.75` |
| `personality_processing` | `false` |
| `ha_base_url` | `` |
| `ha_webhook_enabled` | `false` |
| `stt_provider` | `local` |
| `stt_model` | `ibm-granite/granite-4.0-1b-speech` |
| `stt_base_url` | `` |
| `stt_api_key` | `` |
| `stt_prompt` | `<\|audio\|>can you transcribe...` |
| `speaker_id_model` | `voxceleb_resnet34_LM.onnx` |

### View: `interaction_current_state`

Convenience view joining each interaction with its most recent resolution. Not currently used by the API (which applies the same subquery inline), but useful for ad-hoc queries.

---

## API surface

All routes are under `/api/v1`. The static UI is mounted at `/` after all API routes.

| Method | Path | Purpose |
|---|---|---|
| GET | `/speakers` | List enrolled speakers |
| DELETE | `/speakers/{id}` | Delete speaker + reference clips |
| POST | `/speakers/{id}/clear-clips` | Clear reference clips (keep speaker row) |
| GET | `/interactions/pending` | Pending interactions |
| GET | `/interactions/history` | Full log (filterable by speaker/date) |
| GET | `/audio/{interaction_id}` | Serve WAV file |
| POST | `/interactions/{id}/resolve` | Confirm / reject / assign / enrol / dismiss |
| GET | `/settings` | Read operational settings |
| PUT | `/settings` | Update operational settings |
| GET | `/settings/models` | Read model settings |
| PUT | `/settings/models` | Update model settings |
| POST | `/models/reload` | Hot-swap STT + speaker ID models (background task) |
| GET | `/models/status` | Poll reload status: `reloading` / `ready` / `error` |

### Model reload sequence

```
PUT /settings/models   → saves new settings to DB
POST /models/reload    → returns {"status": "reloading"} immediately
                        starts asyncio.create_task(_do_reload())
GET /models/status     → poll until {"status": "ready"} or {"status": "error"}
```

During reload the old models continue serving requests. The swap is atomic at the orchestrator level — `swap_stt()` and `swap_sid()` are simple attribute assignments on `VoicePipelineOrchestrator`.

---

## Scheduled jobs

Both run on the FastAPI event loop via APScheduler:

| Job | Interval | Purpose |
|---|---|---|
| `RetentionJob.run` | Every 6 hours | Purge old pending interactions; delete WAV files for resolved interactions past retention window |
| `SpeakerIdentifier.refresh_cache` | Every 5 minutes | Rebuild embedding cache from DB (also triggered immediately by `force_refresh()`) |

---

## Retention policy

Two independent policies, both configurable in the UI:

**Pending interactions** — rows older than `pending_retention_days` that are still `pending` are deleted entirely (row + audio + embedding files). Setting to `0` disables this purge.

**Resolved audio** — WAV files for `resolved` interactions older than `resolved_retention_days` are deleted; the metadata row and embedding are kept permanently. Setting to `-1` keeps audio indefinitely.

Reference clips (stored in `enrolled_speakers.reference_clips`) are not touched by retention — they are only rotated during auto-confirm or manual enrolment, bounded by `reference_clips_per_speaker`.

---

## Key design decisions

**SQLite over a dedicated time-series store.** Interaction volume in a home environment is low (tens per day). SQLite with WAL mode handles concurrent readers from the API and concurrent writers from the write path without contention. The entire data layer is a single file that backs up trivially.

**Fire-and-forget writes.** `writer.write()` is dispatched with `asyncio.create_task` so the Wyoming `Transcript` event is sent to HA before the WAV file hits disk. A write failure never blocks the voice pipeline.

**torchaudio monkey-patch.** `wespeakerruntime` calls `torchaudio.load()` internally. Rather than pinning to an older torchaudio, we replace `torchaudio.load` with a `soundfile` implementation at startup. This is an explicit workaround for `torchcodec` availability issues in CUDA-enabled builds; removing it would require changes to `wespeakerruntime`.

**Webhook before transcript.** The Wyoming `Transcript` event is written after `WebhookNotifier.notify()` awaits (with a 1-second timeout). This deliberate ordering ensures Personality LLM has the speaker cache populated before HA invokes the conversation agent. The downside is a maximum 1-second latency penalty on each utterance if HA is unreachable.

**Embeddings are not cross-model compatible.** The WeSpeaker embedding space is entirely distinct from any previous SpeechBrain model. Changing the speaker ID model via the UI requires clearing all enrolled speaker clips and re-enrolling.

---

## File layout

```
api/
  main.py              FastAPI application, all HTTP routes, lifespan
pipeline/
  orchestrator.py      Four-stage pipeline wiring
  writer.py            Fire-and-forget interaction write path
  wyoming_server.py    Wyoming TCP server
  webhook.py           HA Personality LLM webhook notifier
  retention.py         Scheduled retention job
  init_db.py           Schema creation and migration
  models/
    __init__.py        Package re-exports
    noise_suppressor.py  DTLN ONNX wrapper
    vad.py             SileroVAD ONNX wrapper
    speaker_id.py      WeSpeaker ECAPA-TDNN wrapper
    stt.py             SpeechToText factory (_LocalSTT / _RemoteSTT)
ui/
  index.html           Single-page management UI
  app.js               Fetch-based API client, tab management
  style.css            CSS custom properties, dark/light/system theme
docs/
  architecture.md      This file
```
