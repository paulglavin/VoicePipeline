# VoicePipeline

A self-hosted voice pipeline for Home Assistant with speaker identification and a management UI.

## Architecture

Home Assistant sends audio via the Wyoming protocol to a local STT service that runs four models in sequence:

```
HA Wyoming client
      ↓ TCP :10300
DTLN ONNX — noise suppression (wet/dry blend, bypassable via DTLN_MIX=0)
      ↓
SileroVAD v5 ONNX — voice activity detection (gates everything downstream)
      ↓
WeSpeaker ECAPA-TDNN ONNX — speaker identification (192-dim cosine similarity)
      ↓
Granite 4.0 1B Speech — speech-to-text (IBM, gated HuggingFace model)
      ↓
SQLite (WAL mode) — interaction log + speaker profiles
      ↓
Management UI — :8000
```

Every utterance is written to a local SQLite database. Low-confidence or unknown-speaker interactions are queued for manual review in the UI.

### Model implementation notes

- **DTLN** — ONNX Runtime, two-stage model, wet/dry blend controlled by `DTLN_MIX`
- **SileroVAD** — uses the `silero-vad` pip package's bundled ONNX and its `OnnxWrapper` class for state management; no runtime download required
- **WeSpeaker** — uses `wespeakerruntime` (ONNX backend); internally calls `torchaudio.load()` which is monkey-patched at startup to use `soundfile` to avoid a `torchcodec` dependency in torchaudio 2.7+
- **Granite STT** — `transformers` / PyTorch backend; CUDA 12.8 build required for RTX 50-series (Blackwell); no viable ONNX migration path at time of writing

---

## Requirements

- Docker + Docker Compose
- nvidia-container-toolkit (GPU used for Granite STT)
- ~3 GB free disk space for the `./models` bind mount (weights downloaded/copied on first run)
- Ports 8000 (UI/API) and 10300 (Wyoming) available on the host

---

## Deployment

### First time

Clone the repo and build the image:

```bash
git clone <repo-url> VoicePipeline
cd VoicePipeline
cp .env.example .env          # or create .env manually — see HuggingFace token section
docker compose build
docker compose up -d
```

> **Portainer users**: use the manual `docker compose build` step above rather than Portainer's built-in build — large CUDA wheel downloads can time out in Portainer's build runner.

First boot downloads and caches model weights to `./models` on the host:

| Model | Size | Source |
|-------|------|--------|
| DTLN ONNX (×2) | ~1.6 MB | GitHub (downloaded by pipeline at startup) |
| SileroVAD ONNX | ~2 MB | Copied from `silero-vad` pip package (no network required) |
| WeSpeaker ECAPA-TDNN | ~50 MB | Downloaded by `wespeakerruntime` on first call |
| Granite 4.0 1B Speech | ~2 GB | HuggingFace (requires `HF_TOKEN`) |

The healthcheck has a 180-second start window to account for first-run downloads. Subsequent starts are fast because `./models` is a host bind mount that survives rebuilds and `docker compose down -v`.

### Subsequent deploys

```bash
git pull
docker compose build
docker compose up -d
```

### HuggingFace token

`ibm-granite/granite-4.0-1b-speech` is a gated model. Create a `.env` file alongside `docker-compose.yml` before the first run:

```
HF_TOKEN=hf_your_token_here
```

---

## Home Assistant integration

1. In HA: **Settings → Devices & Services → Add Integration → Wyoming Protocol**
2. Host: your server's IP address
3. Port: `10300`

HA will use this as a speech-to-text provider in your assist pipeline.

---

## Enrolling speakers

There is no pre-seeded speaker data. The bootstrap workflow is:

1. Speak to Home Assistant — the interaction appears in the **Pending** tab as _Unknown speaker_
2. Click **Other match → Enrol as new speaker…** and type the person's name
3. Repeat until each household member has been enrolled

Once a speaker has enough high-confidence interactions (≥0.85), their reference clips rotate automatically and identification improves over time.

> **After a container rebuild**: if the speaker ID model has been updated, existing reference clips are not compatible with the new embedding space. Use **Clear clips** in the Enrolled Speakers tab and re-enrol.

---

## Management UI

Open `http://<host>:8000` in a browser.

| Tab | Purpose |
|-----|---------|
| **Pending** | Review unrecognised or low-confidence interactions. Confirm, reject, assign to a different speaker, or enrol a new one. |
| **Enrolled Speakers** | Stats per speaker: interaction count, average confidence, reference clip count, last active. Delete a speaker or clear their reference clips. |
| **History** | Full interaction log, filterable by speaker and date range. |
| **Settings** | Retention policy, speaker matching thresholds, and Home Assistant integration options. |

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `/data/speaker.db` | SQLite database path |
| `AUDIO_DIR` | `/data/audio` | WAV clip storage |
| `EMBEDDING_DIR` | `/data/embeddings` | Speaker embedding cache |
| `MODEL_DIR` | `/models` | ML model weight cache (bind-mounted to `./models`) |
| `WYOMING_HOST` | `0.0.0.0` | Wyoming server bind address |
| `WYOMING_PORT` | `10300` | Wyoming server port |
| `HF_TOKEN` | _(required)_ | HuggingFace token for gated Granite model |
| `DTLN_MIX` | `0.5` | Noise suppression wet/dry blend. `1.0` = full suppression, `0.0` = bypass. Reduce if you hear artefacts (hiss/squelch) on clean audio. |

---

## Data volumes

| Mount | Contents |
|-------|---------|
| `./data` (bind mount) | Database, audio clips, embeddings — **back this up** |
| `./models` (bind mount) | Downloaded model weights — can be deleted and re-downloaded; survives `docker compose down -v` |

---

## Retention

Configured in the **Settings** tab or directly in the database:

| Setting | Default | Description |
|---------|---------|-------------|
| `pending_retention_days` | 7 | Unresolved interactions purged after N days. `0` = never. |
| `resolved_retention_days` | 3 | Audio files for resolved interactions deleted after N days. Metadata kept permanently. `-1` = keep forever. |
| `reference_clips_per_speaker` | 5 | Maximum reference clips retained per speaker (oldest rotated out). |

---

## Personality Engine (HA custom component)

The optional [`ha_agent/`](ha_agent/README.md) component adds per-speaker personality to Home Assistant Assist. Rather than prefixing the transcript, it operates out-of-band:

1. VoicePipeline POSTs speaker identity to an HA webhook immediately after identification
2. The Personality Engine conversation agent reads that cached identity when the Wyoming transcript arrives
3. All utterances are routed through a per-speaker LLM — HA's built-in intent handler executes control commands, and the LLM generates a personality-flavoured spoken response regardless

See [ha_agent/README.md](ha_agent/README.md) for installation and configuration.

---

## Known limitations and future work

- **Granite STT is PyTorch/CUDA only** — no ONNX migration path available yet; this is the main reason `torch` remains a dependency
- **No authentication** — relies on network boundary (home LAN only); not suitable for internet-exposed deployments
- **Speaker matching threshold is a blunt instrument** — a single global threshold (default 0.50) applies to all speakers; per-speaker thresholds would improve accuracy for household members with similar voice characteristics
- **torchaudio 2.7 torchcodec workaround** — `wespeakerruntime` calls `torchaudio.load()` internally; torchaudio 2.7 routes this through `torchcodec` by default, which is unreliable in CUDA builds; the pipeline monkey-patches `torchaudio.load` with a `soundfile` implementation at startup
