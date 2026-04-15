# Personality Engine — HA Custom Component

A speaker-aware personality and TTS routing agent for Home Assistant Assist.

## Overview

Personality Engine gives each household member their own voice: a custom LLM personality, a specific TTS voice, and responses routed to the correct speaker in the room. It works alongside the [VoicePipeline](../README.md) container, which handles noise suppression, VAD, and speaker identification.

**What it does:**

- Receives speaker identity from VoicePipeline via a lightweight webhook
- Selects the per-speaker personality prompt, LLM provider, and TTS voice
- Tries Home Assistant's built-in intent engine first (so lights, locks, and scenes still work)
- Falls back to a per-user LLM (local Ollama or cloud Anthropic/OpenAI) for conversational queries
- Routes TTS output to the media_player associated with the satellite that heard the command

**What it does NOT do:**

- It does not identify speakers — that is VoicePipeline's job
- It does not modify the STT transcript — Wyoming delivers clean text
- It does not require cloud services — Ollama + Piper TTS work fully offline

---

## Installation

1. Copy the `custom_components/personality_engine/` directory into your HA config:

   ```
   config/
   └── custom_components/
       └── personality_engine/
           ├── __init__.py
           ├── manifest.json
           └── ...
   ```

2. Restart Home Assistant.

3. Go to **Settings → Devices & Services → Add Integration** and search for **Personality Engine**.

4. Click through the one-step confirmation. The integration is now active.

5. Click **Configure** on the integration card to add speaker profiles.

---

## Configuration

All configuration is done via the UI. No YAML is needed.

### Adding a speaker

1. **Settings → Devices & Services → Personality Engine → Configure**
2. Choose **Add speaker**
3. Enter the Speaker ID (must match what VoicePipeline sends — e.g. `paul`)
4. Set display name, personality prompt, LLM, and TTS preferences

### Example configurations

**Paul — local Ollama, Piper TTS:**
```
Speaker ID:    paul
Display name:  Paul
System prompt: You are Paul's personal assistant. Paul is technical and prefers
               concise answers. Keep responses under 2 sentences.
LLM provider:  Ollama
Model:         llama3.2:3b
API base:      http://localhost:11434
TTS engine:    tts.piper
TTS voice:     en_US-ryan-medium
```

**Cassandra — Anthropic, cloud TTS:**
```
Speaker ID:    cassandra
Display name:  Cassandra
System prompt: You are Cassandra's helpful home assistant. Cassandra prefers
               friendly, warm responses.
LLM provider:  Anthropic
Model:         claude-sonnet-4-6
API key:       (your Anthropic API key)
TTS engine:    tts.nabu_casa_cloud
TTS voice:     en-GB-SoniaNeural
```

**default — fallback for unrecognised speakers:**
```
Speaker ID:    default
Display name:  Guest
System prompt: You are a helpful home assistant. Keep responses under 3 sentences.
LLM provider:  Ollama
Model:         llama3.2:3b
```

---

## VoicePipeline Integration

### Required changes to VoicePipeline

**1. Add `HA_BASE_URL` to `docker-compose.yml`:**

```yaml
environment:
  HA_BASE_URL: http://homeassistant.local:8123
```

Or in the management UI under Settings → HA Integration.

**2. Enable the HA webhook** in the VoicePipeline Settings tab.

That's it. VoicePipeline will automatically POST speaker metadata to
`http://homeassistant.local:8123/api/webhook/personality_engine_input`
after each speaker identification, and send clean (unprefixed) transcripts
via Wyoming.

### Webhook payload

VoicePipeline sends this after every speaker identification:

```json
{
  "speaker_id":     "paul",
  "confidence":     0.92,
  "timestamp":      "2026-04-15T12:34:56.789Z",
  "interaction_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

The webhook must arrive before the Wyoming transcript reaches HA Assist
(typically 100–500 ms earlier). The agent caches entries for 5 seconds.

### Timing

```
Pipeline processes audio (NS + VAD + speaker ID + STT)
  ↓
POST webhook to HA     ← happens before Wyoming transcript is sent
  ↓
Send Wyoming transcript
  ↓
HA Assist receives transcript, invokes PersonalityEngine
  ↓
PersonalityEngine reads most recent cache entry (≤2 s window)
```

---

## How It Works

### Full flow

```
HA Assist pipeline
  ↓ (Wyoming transcript arrives)
PersonalityConversationAgent.async_process()
  │
  ├─ Read speaker_id from webhook cache (2 s window)
  ├─ Load user config (personality, LLM, TTS)
  ├─ Resolve satellite device_id → satellite entity → media_player
  │
  ├─ Heuristic: looks like a control command?
  │   ├─ YES → delegate to built-in "homeassistant" agent
  │   │         ├─ Intent matched → return HA response (HA handles TTS)
  │   │         └─ No match      → fall through to LLM
  │   └─ NO  → go straight to LLM
  │
  ├─ LLM: route_to_llm(provider, model, system_prompt, text)
  │         └─ On failure → try fallback provider
  │
  ├─ TTS: synthesize_speech(message, tts_config, media_player)
  └─ Return ConversationResult
```

### satellite_id and TTS routing

Home Assistant provides `device_id` in `ConversationInput` — the HA device
registry ID of the satellite. The component resolves this to:

1. The `assist_satellite` entity for that device
2. The associated `media_player` entity (by name, attribute, or area lookup)

This means TTS is automatically delivered to the room the command came from,
with no manual routing configuration needed.

### Speaker cache

The webhook cache uses timestamped keys (`speaker_{ms}_{id}`) so concurrent
requests from multiple satellites never overwrite each other. The agent picks
the most recent entry within a 2-second window.

### conversation_id

The `conversation_id` from `ConversationInput` is passed through to
`ConversationResult` unchanged, maintaining session continuity with HA Assist.
Phase 2: use it to maintain per-session LLM conversation history.

---

## Troubleshooting

**Webhook not received**

- Check VoicePipeline logs for `POST webhook` entries
- Verify `HA_BASE_URL` is correct and HA is reachable from the pipeline container
- Check HA logs for `personality_engine_input` webhook entries
- Confirm the integration is loaded: **Developer Tools → Template** → `{{ integration_loaded('personality_engine') }}`

**Speaker always defaults to "default"**

- The webhook must arrive within 2 seconds before the transcript
- Check pipeline logs: `POST /api/webhook/personality_engine_input` should appear before `Wyoming: Transcript sent`
- Verify the `speaker_id` in the webhook matches the profile name in the options flow (case-insensitive)

**LLM timeout**

- Ollama: check `ollama serve` is running and the model is pulled (`ollama pull llama3.2:3b`)
- Anthropic/OpenAI: check API key is valid and not rate-limited
- Increase `max_tokens` if responses are being cut short
- Configure a fallback provider for resilience

**TTS not playing**

- Check HA logs for `tts.speak` service call errors
- Verify the TTS engine entity exists (`tts.piper`, etc.)
- Use **Developer Tools → Services** to test `tts.speak` manually
- Check satellite → media_player resolution in logs (set HA logger to DEBUG for `custom_components.personality_engine`)

**Local intents stop working**

- The agent tries HA intents first for control commands — this should be transparent
- If a control phrase is being sent to the LLM, check the intent heuristic keywords in `conversation.py:_CONTROL_KEYWORDS`
- Add the phrase pattern to `_CONTROL_KEYWORDS` if needed

---

## Architecture

### Responsibilities

| Responsibility | Owner |
|----------------|-------|
| Noise suppression | VoicePipeline (DTLN ONNX) |
| Voice activity detection | VoicePipeline (SileroVAD) |
| Speaker identification | VoicePipeline (WeSpeaker ECAPA-TDNN) |
| Speech-to-text | VoicePipeline (Granite 4.0 1B) |
| Speaker metadata delivery | VoicePipeline → HA webhook |
| Personality selection | Personality Engine (this component) |
| LLM routing | Personality Engine |
| TTS routing | Personality Engine |
| Device/satellite context | Home Assistant (satellite_id, conversation_id) |
| Intent processing | Home Assistant built-in agent |

### Key design decisions

**Webhook not transcript prefix** — speaker identity arrives out-of-band so the
transcript is always clean text. This allows HA's intent matcher to work without
modification, and removes the coupling between pipeline and HA format.

**Local intent first** — the built-in HA agent handles lights/locks/scenes.
The LLM is only invoked for conversational queries. This keeps latency low for
the 90% case and preserves intent script behaviour.

**Per-user LLM** — each speaker can have a different LLM provider and model.
A privacy-conscious user can use local Ollama while another uses Claude.

**No device detection in pipeline** — satellite_id comes from HA's Assist
context. The pipeline never needs to know which device heard the audio.
