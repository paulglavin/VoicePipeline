# Integration Contract — VoicePipeline ↔ Personality Engine

This document defines the interface between the VoicePipeline container and
the Personality Engine HA custom component. Both sides must conform to this
contract for the integration to work correctly.

---

## Data flow

```
┌─────────────────────────────────────────────────────────────────┐
│  VoicePipeline container                                        │
│                                                                 │
│  audio → DTLN → SileroVAD → WeSpeaker → Granite STT            │
│                                  │              │               │
│                          speaker_id        transcript           │
│                          confidence             │               │
│                               │                │               │
│                    POST /api/webhook/  Wyoming TCP :10300       │
│                    personality_engine_input                     │
└───────────────────────────────┼────────────────┼───────────────┘
                                │                │
                    ────────────┘                └────────────
                    │                                        │
         arrives first (before transcript)           arrives second
                    │                                        │
┌───────────────────┼────────────────────────────────────────┼───┐
│  Home Assistant                                             │   │
│                   ↓                                         ↓   │
│  Personality Engine webhook handler      HA Assist pipeline     │
│  → stores in speaker cache          → invokes conversation agent│
│                                                                 │
│  PersonalityConversationAgent.async_process()                   │
│  ├─ reads speaker from cache (≤2 s window)                      │
│  ├─ selects user config                                         │
│  ├─ tries local intent (built-in HA agent)                      │
│  ├─ falls back to per-user LLM                                  │
│  └─ routes TTS to media_player via satellite_id                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Webhook specification

### Endpoint

```
POST /api/webhook/personality_engine_input
```

HA URL: `http://<ha-host>:8123/api/webhook/personality_engine_input`

No authentication required (HA webhooks are public within the network).

### Request

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
  "speaker_id":     "paul",
  "confidence":     0.92,
  "timestamp":      "2026-04-15T12:34:56.789Z",
  "interaction_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Field definitions:**

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `speaker_id` | string | yes | `[a-z0-9_]{1,50}` | Speaker name (lowercase). Must match the profile key in Personality Engine config. |
| `confidence` | number | yes | 0.0–1.0 | WeSpeaker cosine similarity score. |
| `timestamp` | string | yes | ISO-8601 UTC | When identification occurred. |
| `interaction_id` | string | no | UUID format | VoicePipeline interaction ID. Used for log correlation. |

**Notes:**
- `speaker_id` must be the exact lowercase name as enrolled in VoicePipeline (e.g. `paul`, not `Paul`).
- If no speaker matched (unknown voice), do **not** send a webhook. The Personality Engine will use the `default` profile.
- If speaker matched below the match threshold, send the webhook with the best-match `speaker_id` and its `confidence`. The Personality Engine decides whether to trust it.

### Response

**200 OK (success):**
```json
{"success": true}
```

**400 Bad Request (invalid JSON):**
```json
{"success": false, "error": "invalid JSON"}
```

**422 Unprocessable (validation failure):**
```json
{"success": false, "error": "invalid speaker_id"}
```

VoicePipeline should log non-200 responses as warnings but continue normally.

### Timing

The webhook **must** be sent after speaker identification but **before** the
Wyoming `Transcript` event is written. The Personality Engine caches entries
for 5 seconds; the conversation agent reads the cache within a 2-second window
after the transcript arrives.

Typical timing budget:

```
t=0       AudioStop received by VoicePipeline
t=50 ms   Noise suppression complete
t=80 ms   VAD complete
t=150 ms  Speaker identification complete
t=155 ms  Webhook POST fired (awaited, timeout=1 s)
t=200 ms  Webhook response received from HA
t=1500 ms STT complete
t=1505 ms Wyoming Transcript sent to HA
t=1600 ms HA invokes PersonalityConversationAgent
t=1600 ms Agent reads cache (entry is 1450 ms old — within 2 s window ✓)
```

### Timeout behaviour

If HA is unreachable or the webhook takes >1 second:
- VoicePipeline logs a warning and continues
- Wyoming transcript is sent without a preceding webhook
- Personality Engine uses the `default` profile for that utterance

---

## Wyoming protocol

The Wyoming interface is **unchanged** from the standard HA STT integration.

| Parameter | Value |
|-----------|-------|
| Protocol | Wyoming TCP |
| Port | 10300 |
| Audio format | 16-bit PCM, 16 kHz, mono |
| Transcript format | Plain text, **no speaker prefix** |

VoicePipeline must **not** include a speaker prefix in the transcript
(e.g. `"Paul: turn the lights off"` is wrong; `"turn the lights off"` is correct).

---

## Responsibilities matrix

| Concern | VoicePipeline | Personality Engine |
|---------|---------------|--------------------|
| Noise suppression | ✓ | |
| Voice activity detection | ✓ | |
| Speaker identification | ✓ | |
| Speech-to-text | ✓ | |
| Speaker metadata delivery (webhook) | ✓ | |
| Transcript delivery (Wyoming) | ✓ | |
| Satellite/device context | | ✓ (from HA Assist) |
| TTS routing | | ✓ |
| Per-speaker personality | | ✓ |
| LLM invocation | | ✓ |
| Intent processing | | ✓ (delegates to HA) |
| Speaker profile management UI | ✓ | |

---

## Versioning

This contract is **v1**. Breaking changes require a version bump and must be
documented in both `VOICEPIPELINE_CHANGES.md` and this file.

Breaking changes are:
- Renaming or removing required webhook fields
- Changing the webhook endpoint path
- Changing the Wyoming transcript format

Non-breaking changes (additive):
- Adding optional webhook fields
- Adding new HA component features
