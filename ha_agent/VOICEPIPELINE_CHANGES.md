# VoicePipeline Changes Required for Personality Engine

Two changes are required in the VoicePipeline container:

1. **Add webhook integration** — POST speaker metadata to HA after each identification
2. **Remove transcript prefixing** — always send clean text via Wyoming

---

## 1. Add Webhook Integration

### What changed

After speaker identification, VoicePipeline now POSTs a JSON payload to HA's
webhook endpoint before sending the Wyoming transcript. This gives the
Personality Engine the speaker's identity before it processes the utterance.

### Configuration

Add `HA_BASE_URL` to `docker-compose.yml`:

```yaml
environment:
  HA_BASE_URL: "http://homeassistant.local:8123"
  HA_WEBHOOK_ENABLED: "true"
```

Or configure via the VoicePipeline management UI: **Settings → HA Integration**.

If `HA_BASE_URL` is empty or `HA_WEBHOOK_ENABLED` is `false`, the webhook is
silently skipped and the pipeline behaves as before (speaker unknown to HA).

### Webhook endpoint

```
POST http://<HA_BASE_URL>/api/webhook/personality_engine_input
Content-Type: application/json
```

### Payload

```json
{
  "speaker_id":     "paul",
  "confidence":     0.92,
  "timestamp":      "2026-04-15T12:34:56.789Z",
  "interaction_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `speaker_id` | string | yes | Lowercase speaker name from enrolled_speakers |
| `confidence` | float | yes | Match confidence 0.0–1.0 |
| `timestamp` | ISO-8601 string | yes | UTC timestamp of identification |
| `interaction_id` | UUID string | no | VoicePipeline interaction ID for correlation |

### Timing requirement

The webhook POST must complete **before** `Transcript` is written to the
Wyoming stream. The pipeline awaits the POST with a 1-second timeout —
if HA is unreachable, the transcript is still sent (with no speaker context).

### Code location

- `pipeline/webhook.py` — `WebhookNotifier` class
- `pipeline/wyoming_server.py` — calls `WebhookNotifier.notify()` before writing Transcript

---

## 2. Remove Transcript Prefixing

### What changed

Previously, when `personality_processing` was enabled in Settings, the
pipeline prepended the speaker name to the transcript:

```
"Paul: turn the kitchen lights off"
```

This prefix has been **removed**. Wyoming now always sends clean text:

```
"turn the kitchen lights off"
```

### Why

The prefix was a workaround for passing speaker identity through the Wyoming
protocol, which carries no metadata. With the webhook providing speaker
identity out-of-band, the prefix is no longer needed and would break HA's
intent matcher.

### Migration

If you have any automations or intent scripts that pattern-match on the
`"Paul: ..."` prefix format, update them to use the Personality Engine's
per-speaker routing instead.

The `personality_processing` setting in the management UI now controls
whether the webhook is sent (rather than prefix injection). The setting name
is preserved for backwards compatibility but its behaviour has changed.

---

## 3. Configuration reference

New environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HA_BASE_URL` | `` (empty) | Home Assistant base URL. Webhook disabled if empty. |
| `HA_WEBHOOK_ENABLED` | `false` | Explicit enable flag (overrides HA_BASE_URL check). |

New database settings (settable via management UI):

| Key | Default | Description |
|-----|---------|-------------|
| `ha_base_url` | `` | HA base URL (persisted, overrides env var at runtime). |
| `ha_webhook_enabled` | `false` | Enable webhook posting. |

---

## 4. Testing checklist

- [ ] Rebuild container after adding `HA_BASE_URL` to `docker-compose.yml`
- [ ] Speak a command; check VoicePipeline logs for `Webhook: POST → HA succeeded`
- [ ] Check HA logs for `Personality Engine webhook: cached paul conf=0.92`
- [ ] Verify transcript arrives in HA **without** the `"Paul: "` prefix
- [ ] Confirm Personality Engine selects the correct speaker personality
- [ ] Test with HA unreachable: pipeline should continue normally (webhook timeout logged as warning, not error)
- [ ] Test with unknown speaker: `speaker_id` absent means cache miss → "default" personality used
