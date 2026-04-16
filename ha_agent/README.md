# Personality Engine — HA Custom Component

A speaker-aware personality conversation agent for Home Assistant Assist.

## Overview

Personality Engine gives each household member their own voice: a custom LLM personality and per-speaker configuration. It works alongside the [VoicePipeline](../README.md) container, which handles noise suppression, VAD, speaker identification, and STT.

**What it does:**

- Receives speaker identity from VoicePipeline via a lightweight webhook (out-of-band, before the transcript arrives)
- Selects the per-speaker personality prompt and LLM provider
- Executes HA control commands via the built-in intent handler (lights, locks, scenes, etc.)
- Routes every utterance through a per-speaker LLM — even control commands get a personality-flavoured spoken response
- Returns a `ConversationResult`; HA's Assist pipeline handles TTS delivery to the satellite

**What it does NOT do:**

- It does not identify speakers — that is VoicePipeline's job
- It does not modify the STT transcript — Wyoming delivers clean text to HA
- It does not call HA services directly — control commands are delegated to HA's built-in intent handler
- It does not handle TTS routing — HA's Assist pipeline sends TTS to the satellite that heard the command

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

2. Delete any existing `__pycache__` directories inside `personality_engine/` if updating.

3. Restart Home Assistant fully (`ha core restart` from the terminal, not the UI reload).

4. Go to **Settings → Devices & Services → Add Integration** and search for **Personality Engine**.

5. Click through the one-step confirmation. The integration is now active.

6. Click **Configure** to add speaker profiles.

---

## Assist Pipeline Setup

Two settings must be correct for the agent to work end-to-end.

### 1. Select Personality Engine as the conversation agent

**Settings → Voice Assistants → [your pipeline] → Conversation agent → Personality Engine**

### 2. Disable "Prefer local intents" on the satellite

**Settings → Devices → [satellite device] → Configure → Prefer local intents → OFF**

When this is on, HA's Assist pipeline handles matching control commands internally and never calls the Personality Engine's `async_process`. Turning it off routes every utterance through the agent, which then delegates control commands to the built-in intent handler internally.

---

## Configuration

All configuration is done via the UI. No YAML is needed.

### Adding a speaker

1. **Settings → Devices & Services → Personality Engine → Configure**
2. Choose **Add speaker**
3. Enter the Speaker ID — this must exactly match the `speaker_id` field VoicePipeline sends in the webhook (e.g. `paul`)
4. Set display name, personality prompt, LLM provider, and optional fallback

### Fields

| Field | Description |
|-------|-------------|
| Speaker ID | Lowercase, alphanumeric + underscores. Must match VoicePipeline's enrolled speaker name exactly. |
| Display name | Used in the LLM system prompt ("The user's name is Paul") |
| System prompt | Persona instructions for the LLM |
| LLM provider | `ollama`, `anthropic`, `openai`, or `llamacpp` |
| Model | Model identifier for the primary provider |
| API base | Base URL for the provider (Ollama/llama.cpp only; leave blank for cloud providers) |
| API key | API key (Anthropic/OpenAI; leave blank for local providers) |
| Fallback provider/model | Used if the primary provider fails |
| Fallback API base/key | Credentials for the fallback provider if different from primary |

### Example configurations

**Paul — llama.cpp, J.A.R.V.I.S. personality:**
```
Speaker ID:    paul
Display name:  Paul
System prompt: You are J.A.R.V.I.S., Tony Stark's AI from Iron Man.
               You address Paul as "sir". Keep responses to one or two sentences.
LLM provider:  llama.cpp
Model:         GPTOSS-2OB
API base:      http://192.168.2.173:11434
Fallback:      anthropic / claude-sonnet-4-6
Fallback key:  (your Anthropic API key)
```

**default — fallback for unrecognised speakers:**
```
Speaker ID:    default
Display name:  Guest
System prompt: You are a helpful home assistant. Keep responses under 3 sentences.
LLM provider:  llamacpp
Model:         GPTOSS-2OB
API base:      http://192.168.2.173:11434
```

The `default` profile is used when VoicePipeline does not identify the speaker or the webhook cache has expired. Configure it with a working LLM to avoid failures for unrecognised speakers.

---

## VoicePipeline Integration

### docker-compose.yml

Add these environment variables to the VoicePipeline service:

```yaml
environment:
  HA_BASE_URL: "http://192.168.2.30:8123"   # your HA IP
  HA_WEBHOOK_ENABLED: "true"
```

Environment variables take priority over the Settings UI values.

### How the webhook works

VoicePipeline POSTs speaker metadata to HA immediately after speaker identification, before the Wyoming transcript is sent:

```json
{
  "speaker_id":     "paul",
  "confidence":     0.92,
  "timestamp":      "2026-04-16T12:34:56.789Z",
  "interaction_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

The Personality Engine caches this entry. When the Wyoming transcript arrives moments later and HA invokes `async_process`, the agent reads the most recent cache entry within a 2-second window.

---

## How It Works

### Full flow per utterance

```
HA Assist pipeline receives Wyoming transcript
  ↓
PersonalityConversationAgent.async_process()
  │
  ├─ Read speaker_id from webhook cache (2 s window)
  │    └─ No entry → speaker_id = "default"
  │
  ├─ Load per-speaker user config (personality, LLM provider/model)
  │
  ├─ Heuristic: does this look like a control command?
  │   └─ YES → delegate to conversation.home_assistant (built-in intent handler)
  │              ├─ Intent matched → ha_action_result = HA's confirmation text
  │              └─ No match      → ha_action_result = None
  │
  ├─ LLM call (always)
  │    ├─ system_prompt = personality + "user's name is {display_name}"
  │    ├─ if ha_action_result: append "HA already executed this: {result}. Confirm in character."
  │    ├─ user_message = original utterance
  │    └─ On failure → try fallback provider → on failure → return ha_action_result or error string
  │
  └─ Return ConversationResult(speech=llm_response)
       └─ HA Assist pipeline delivers TTS to the satellite
```

The key point: **the LLM always runs**. For control commands, the action executes via HA's intent handler and the LLM confirms it in character. For conversational queries, the LLM answers directly. The spoken response always has the speaker's personality.

### Why the LLM always runs

HA's built-in intent handler returns generic confirmations ("Turned on the lights"). The Personality Engine replaces these with personality-flavoured responses by always routing through the LLM, passing the HA confirmation as context so the LLM knows what was done.

### Tool calling limitation

The current architecture cannot handle commands that require HA service calls beyond the basic intent handler — timers, complex scenes, state queries. For full tool calling support, configure Extended OpenAI Conversation (HACS) or HA's built-in LLM conversation integration, and either:

- Use it as the conversation agent with a personality-configured system prompt (recommended, single LLM call)
- Set it as the execution delegate in `_try_local_intent` instead of `conversation.home_assistant`

---

## Supported LLM Providers

| Provider | Key | Notes |
|----------|-----|-------|
| Ollama | `ollama` | Local. Default port 11434. |
| llama.cpp | `llamacpp` | Local. OpenAI-compatible `/v1/chat/completions`. Any port. For reasoning models, set `"tool_choice": "none"` is sent automatically to prevent spurious tool call attempts. |
| Anthropic | `anthropic` | Cloud. Requires `api_key`. |
| OpenAI | `openai` | Cloud. Requires `api_key`. Also compatible with Azure OpenAI via `api_base`. |

### Reasoning models (llama.cpp)

Models with `reasoning_effort` configured server-side (e.g. GPT-OSS, Qwen3 reasoning variants) may return `content: null` with the response in `reasoning_content`. The provider handles this automatically and falls back to `reasoning_content` if `content` is empty.

---

## Troubleshooting

**No personality response — plain "Turned off the lights" spoken**

- Check that **Prefer local intents** is disabled on the satellite. When enabled, HA handles matching commands before calling the Personality Engine.
- Verify **Personality Engine** is selected as the conversation agent in the Assist pipeline.

**"Sorry, I'm having trouble thinking right now"**

- The LLM call failed. Check HA logs (**Settings → System → Logs**, filter `personality_engine`) for the specific error.
- Common causes: wrong `api_base`, model name mismatch, provider unreachable, missing API key for cloud providers.
- Verify the primary and fallback provider configs in the speaker profile.

**Speaker always defaults to "default"**

- The webhook must arrive before the transcript. Check VoicePipeline logs for `Webhook: POST → HA succeeded`.
- Verify `HA_BASE_URL` is set and reachable from the container.
- Check that the speaker ID in the Personality Engine config matches exactly what VoicePipeline sends (shown in pipeline logs: `speaker=paul`).

**"Agent homeassistant not found" in logs (local intent delegation)**

- The local intent delegation uses `agent_id="conversation.home_assistant"`. If that agent is not available, local intent execution is skipped silently and the LLM responds without executing the action.
- The lights will not actually change in this case. Ensure HA's built-in conversation integration is enabled.

**New files not loading after copy**

1. Delete `/config/custom_components/personality_engine/__pycache__/`
2. Run `ha core restart` (full process restart — not the UI reload button)
3. Check for `PersonalityEngine v2: async_process called` WARNING in System Log after speaking

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
| Personality and LLM routing | Personality Engine (this component) |
| HA control command execution | HA built-in intent handler (delegated) |
| TTS delivery | HA Assist pipeline (satellite → TTS engine) |

### Key design decisions

**Webhook, not transcript prefix** — speaker identity arrives out-of-band so the transcript is always clean text. HA's intent matcher works without modification and there is no coupling between pipeline output format and HA's parsing.

**LLM always runs** — every utterance gets a personality-flavoured response. Control commands are executed by HA's intent handler first; the LLM confirms what was done in character rather than generating a generic "Turned on the lights".

**`prefer_local_intents: false`** — when this is on, HA bypasses the conversation agent entirely for matched intents. It must be off so the Personality Engine receives every utterance.

**Per-speaker LLM** — each speaker can have a different provider and model. A privacy-conscious user can use a local llama.cpp model while another uses Claude.

**Fallback credentials are independent** — fallback provider has its own `api_base` and `api_key`, so a fallback to Anthropic (cloud) works correctly when the primary is a local provider.
