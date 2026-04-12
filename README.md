# VoicePipeline

A self-hosted voice pipeline for Home Assistant with speaker identification and a management UI.

## Architecture

Home Assistant sends audio via the Wyoming protocol to a local STT service that runs four models in sequence:

```
HA Wyoming client
      ↓ TCP :10300
GTCRN — noise suppression
      ↓
SileroVAD — voice activity detection
      ↓
ECAPA-TDNN — speaker identification
      ↓
Granite 4.0 1B Speech — speech-to-text
      ↓
SQLite — interaction log
      ↓
Management UI — :8000
```

Every utterance is written to a local SQLite database. Low-confidence or unknown-speaker interactions are queued for manual review in the UI.

## Requirements

- Docker + Docker Compose
- ~5 GB free disk space (model weights downloaded on first run)
- Ports 8000 (UI/API) and 10300 (Wyoming) available on the host

## Deployment

### First time

Clone the repo and build the image manually — Portainer's built-in build step does not handle large builds reliably:

```bash
git clone <repo-url> VoicePipeline
cd VoicePipeline
docker compose build
```

Then deploy the stack via Portainer (or `docker compose up -d`).

First boot downloads ~3–4 GB of model weights to the `model_cache` Docker volume. The healthcheck has a 180-second start window to account for this. Subsequent starts are fast.

### Subsequent deploys

```bash
git pull
docker compose build
docker compose up -d
```

Portainer can manage the running stack after the initial manual build.

### HuggingFace token

If `ibm-granite/granite-speech-4.0-1b` is a gated model, uncomment and set the token in `docker-compose.yml` before building:

```yaml
environment:
  HUGGING_FACE_HUB_TOKEN: hf_your_token_here
```

## Home Assistant integration

1. In HA: **Settings → Devices & Services → Add Integration → Wyoming Protocol**
2. Host: your server's IP address
3. Port: `10300`

HA will use this as a speech-to-text provider in your assist pipeline.

## Enrolling speakers

There is no pre-seeded speaker data. The bootstrap workflow is:

1. Speak to Home Assistant — the interaction appears in the **Pending** tab as _Unknown speaker_
2. Click **Other match → Enrol as new speaker…** and type the person's name
3. Repeat until each household member has been enrolled

Once a speaker has enough high-confidence interactions (≥0.85), their reference clips rotate automatically and identification improves over time.

## Management UI

Open `http://<host>:8000` in a browser.

| Tab | Purpose |
|-----|---------|
| **Pending** | Review unrecognised or low-confidence interactions. Confirm, reject, assign to a different speaker, or enrol a new one. |
| **Enrolled Speakers** | Stats per speaker: interaction count, average confidence, reference clip count, last active. |
| **History** | Full interaction log, filterable by speaker and date range. |
| **Settings** | Retention policy for pending and resolved interactions. |

## Environment variables

All have sensible defaults; override in `docker-compose.yml` as needed.

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `/data/speaker.db` | SQLite database path |
| `AUDIO_DIR` | `/data/audio` | WAV clip storage |
| `EMBEDDING_DIR` | `/data/embeddings` | Speaker embedding cache |
| `MODEL_DIR` | `/models` | ML model weight cache |
| `WYOMING_HOST` | `0.0.0.0` | Wyoming server bind address |
| `WYOMING_PORT` | `10300` | Wyoming server port |

## Data volumes

| Volume | Contents |
|--------|---------|
| `./data` (bind mount) | Database, audio clips, embeddings — back this up |
| `model_cache` (named) | Downloaded model weights — can be deleted and re-downloaded |

## Retention

Configured in the **Settings** tab or directly in the database:

| Setting | Default | Description |
|---------|---------|-------------|
| `pending_retention_days` | 7 | Unresolved interactions purged after N days. `0` = never. |
| `resolved_retention_days` | 3 | Audio files for resolved interactions deleted after N days. Metadata kept permanently. `-1` = keep forever. |
| `reference_clips_per_speaker` | 5 | Maximum reference clips retained per speaker (oldest rotated out). |
