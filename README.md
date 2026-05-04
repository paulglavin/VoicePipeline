# VoicePipeline

A self-hosted voice pipeline for Home Assistant that adds speaker identification to your Assist setup — so your assistant knows who's talking and can respond accordingly.

Audio from Home Assistant travels through a four-stage ML pipeline: noise suppression → voice activity detection → speaker identification → speech-to-text. Every utterance is stored locally, and a management UI lets you review low-confidence detections and enrol household members.

Designed to work alongside the [Personality LLM](https://github.com/PaulGlavin/personality_llm) Home Assistant integration, which uses speaker identity to deliver per-person personality and tone.

---

## How it fits into your HA setup

```
Microphone → Home Assistant Assist pipeline
                    ↓ Wyoming protocol (TCP :10300)
             VoicePipeline
                    ↓ identifies speaker
             fires webhook → Personality LLM (HA integration)
                    ↓ routes to correct user profile
             personalised response spoken back
```

Without Personality LLM, VoicePipeline still works as a standard Wyoming STT provider — it just won't route personality per speaker.

---

## Requirements

- Docker + Docker Compose
- Ports `8000` (management UI) and `10300` (Wyoming) available on the host
- **For local Granite STT only**: NVIDIA GPU with `nvidia-container-toolkit` installed, CUDA 12.8+, ~3 GB free disk space for model weights
  - Not required if using the slim image with a [remote STT endpoint](#configuring-models)

---

## Docker images

Pre-built images are published to the GitHub Container Registry on every push to `main`:

| Image | Size | GPU required | STT |
|-------|------|-------------|-----|
| `ghcr.io/paulglavin/voicepipeline:latest` | ~400 MB | No | Remote endpoint |
| `ghcr.io/paulglavin/voicepipeline:cuda` | ~2.5 GB | Yes (CUDA 12.8+) | Local Granite 4.0 1B |

**Most users should start with `:latest`** — configure a remote STT endpoint in the management UI after first boot. Use `:cuda` only if you want to run Granite on-device.

---

## Installation

### 1. Configure

```bash
git clone https://github.com/PaulGlavin/VoicePipeline
cd VoicePipeline
cp .env.example .env
```

If you plan to use the `:cuda` image with the default Granite STT model, add your HuggingFace token to `.env`:

```
HF_TOKEN=hf_your_token_here
```

Skip this if using `:latest` with a remote STT endpoint.

### 2. Start

```bash
docker compose up -d
```

Docker Compose pulls the pre-built image automatically. The management UI is available at `http://<your-server-ip>:8000`.

> **To use the CUDA image**, edit `docker-compose.yml` and change the image tag from `:latest` to `:cuda`, then add the `nvidia` runtime and `HF_TOKEN` to the service definition.

> **To build locally** (development or custom changes):
> ```bash
> docker compose build
> docker compose up -d
> ```
> For the CUDA variant: `docker compose build --build-arg ENABLE_LOCAL_STT=true`

First boot with the CUDA image downloads model weights to `./models`:

| Model | Size | Notes |
|-------|------|-------|
| DTLN ONNX (×2) | ~2 MB | Noise suppressor — downloaded from GitHub |
| SileroVAD ONNX | ~2 MB | VAD — copied from pip package, no download |
| WeSpeaker ECAPA-TDNN | ~50 MB | Speaker ID — downloaded by wespeakerruntime |
| Granite 4.0 1B Speech | ~2 GB | STT — requires `HF_TOKEN`, downloaded from HuggingFace |

The `./models` directory is a host bind mount that survives image updates.

### 3. Add the Wyoming integration in Home Assistant

1. **Settings → Devices & Services → Add Integration → Wyoming Protocol**
2. Host: your server's IP address
3. Port: `10300`

HA will now use VoicePipeline as its speech-to-text provider.

---

## Enrolling speakers

There is no pre-seeded speaker data. The first-time workflow is:

1. Speak to Home Assistant — the utterance appears in the **Pending** tab as _Unknown speaker_
2. Click **Other match → Enrol as new speaker…** and enter the person's name
3. Repeat for each household member until everyone is enrolled

Once a speaker has accumulated enough high-confidence interactions (≥0.85 similarity), their reference clips rotate automatically and identification improves over time.

Open the management UI at `http://<your-server-ip>:8000`.

> **After a container rebuild involving a speaker ID model change**: existing embeddings are not compatible with a different model's embedding space. Use **Clear clips** in the Enrolled Speakers tab and re-enrol each speaker.

---

## Management UI

`http://<host>:8000`

| Tab | Purpose |
|-----|---------|
| **Pending** | Review unknown or low-confidence interactions. Confirm, reject, assign to a different speaker, or enrol a new one. |
| **Enrolled Speakers** | Stats per speaker — interaction count, average confidence, reference clip count, last active. Delete a speaker or clear their clips. |
| **History** | Full interaction log, filterable by speaker and date range. Expandable rows show pipeline timings. |
| **Settings** | Speaker matching thresholds, retention policy, Home Assistant webhook, and model configuration. |

---

## Configuring models

The **Settings → Speech Recognition** and **Settings → Speaker Identification** sections let you change models without editing code or restarting the container.

### STT: Local vs Remote

| Provider | When to use |
|----------|-------------|
| **Local (HuggingFace)** | Default. Runs on-device; requires a GPU and `HF_TOKEN` for gated models. |
| **Remote (OpenAI-compatible)** | Use when you have a separate GPU machine running Ollama or another OpenAI-compatible server. No local GPU required. |

For remote STT:
1. Select **Remote (OpenAI-compatible)** as the provider
2. Set the **API base URL** (e.g. `http://bigbox:11434/v1` for Ollama)
3. Enter the **model name** as known to that server
4. Clear the **transcription prompt** field (the default is tuned for Granite and won't work on other models)
5. Click **Save & reload models**

### Speaker ID model

The default WeSpeaker VoxCeleb ResNet34 model is downloaded automatically. To use a different ONNX model, place the file in the wespeaker directory inside your `./models` bind mount and enter the filename in the Speaker Identification settings.

---

## Personality LLM integration

[Personality LLM](https://github.com/PaulGlavin/personality_llm) is a Home Assistant custom integration that delivers per-speaker personality from a local LLM. VoicePipeline feeds it speaker identity via a webhook.

### How it works

After identifying a speaker, VoicePipeline fires a webhook to HA:

```
POST /api/webhook/personality_llm_input
{
  "speaker_id": "paul",
  "confidence": 0.91,
  "timestamp": "2026-05-04T09:00:00Z",
  "interaction_id": "..."
}
```

The Personality LLM integration caches this for 2 seconds. When the Wyoming transcript arrives a moment later, it reads the cached speaker identity and routes the conversation to that person's profile — adjusting personality style, humour level, and response format.

### Configuration

1. Install [Personality LLM](https://github.com/PaulGlavin/personality_llm) via HACS
2. In the Personality LLM options, enable **per-user personality** and create a profile for each speaker (use the same names you enrolled in VoicePipeline)
3. In VoicePipeline **Settings → Home Assistant Integration**:
   - Enable the HA webhook
   - Enter your HA base URL and a long-lived access token
4. Select the Personality LLM conversation agent in your HA Assist pipeline

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `/data/speaker.db` | SQLite database path |
| `AUDIO_DIR` | `/data/audio` | WAV clip storage |
| `EMBEDDING_DIR` | `/data/embeddings` | Speaker embedding cache |
| `MODEL_DIR` | `/models` | ML model weight cache |
| `WYOMING_HOST` | `0.0.0.0` | Wyoming server bind address |
| `WYOMING_PORT` | `10300` | Wyoming server port |
| `HF_TOKEN` | _(required for local Granite STT)_ | HuggingFace token for gated models |
| `DTLN_MIX` | `0.5` | Noise suppressor wet/dry blend. `1.0` = full, `0.0` = bypass. Reduce if you hear artefacts on clean audio. |

---

## Data volumes

| Mount | Contents |
|-------|---------|
| `./data` | Database, audio clips, embeddings — **back this up** |
| `./models` | Downloaded model weights — can be deleted and re-downloaded |

---

## Retention

Configured in **Settings → Retention** or directly in the database:

| Setting | Default | Description |
|---------|---------|-------------|
| Pending retention | 7 days | Unresolved interactions purged after N days. `0` = never. |
| Resolved audio retention | 3 days | WAV files deleted after N days. Metadata kept permanently. `-1` = keep forever. |
| Reference clips per speaker | 5 | Maximum clips per speaker — oldest rotated out first. |

---

## Updating

```bash
docker compose pull
docker compose up -d
```

Model weights in `./models` are preserved across image updates.

---

## Known limitations

- **No authentication** — intended for home LAN only; do not expose to the internet
- **Single global match threshold** — one threshold applies to all speakers; household members with similar voices may need manual threshold tuning
- **Local STT requires PyTorch/CUDA** — no ONNX path for Granite; use the remote STT option if you don't have a local GPU

For implementation details see [docs/architecture.md](docs/architecture.md).
