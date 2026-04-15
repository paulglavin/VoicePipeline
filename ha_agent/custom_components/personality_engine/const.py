"""Constants for the Personality Engine integration."""

DOMAIN = "personality_engine"

# Webhook ID registered with HA's webhook component.
# VoicePipeline POSTs speaker metadata here before sending the Wyoming transcript.
WEBHOOK_ID = "personality_engine_input"

# How long (seconds) a speaker cache entry is considered valid.
# Must exceed the worst-case pipeline latency (NS + VAD + speaker ID + STT).
# The Wyoming transcript arrives at HA ~1–3 seconds after the webhook; 5 s
# gives comfortable headroom without matching across utterances.
SPEAKER_CACHE_TTL = 5  # seconds

# How often (seconds) the cache cleanup task runs.
SPEAKER_CACHE_CLEANUP_INTERVAL = 10  # seconds

# ── LLM provider identifiers ──────────────────────────────────────────────────

PROVIDER_OLLAMA = "ollama"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
PROVIDER_LLAMACPP = "llamacpp"

# ── LLM provider recommendations (for config UI and documentation) ────────────
#
# LOCAL / PRIVATE / FAST:
#   provider: ollama
#   model:    llama3.2:3b
#   latency:  200–500 ms; runs on modest hardware; strong instruction following.
#
# CLOUD / HIGH-QUALITY:
#   provider: anthropic
#   model:    claude-sonnet-4-6
#   latency:  500–1200 ms; best reasoning and conversational quality.
#
# BALANCED (recommended starting point):
#   primary:  ollama / llama3.2:3b
#   fallback: anthropic / claude-sonnet-4-6

# ── Storage ───────────────────────────────────────────────────────────────────

STORAGE_KEY = "personality_engine_users"
STORAGE_VERSION = 1

# ── hass.data keys ────────────────────────────────────────────────────────────

DATA_CONFIG_MANAGER = "config_manager"
DATA_SPEAKER_CACHE = "speaker_cache"
DATA_USER_MAP = "user_map"  # speaker_id → HA user_id

# ── Default per-user configuration ───────────────────────────────────────────

DEFAULT_USER_CONFIG: dict = {
    "display_name": "Guest",
    "personality": {
        # System prompt injected before every LLM conversation.
        # Phase 2: load from per-user file / HA template.
        "system_prompt": (
            "You are a helpful, polite home assistant. "
            "Keep responses concise (under 3 sentences)."
        ),
        "temperature": 0.7,
        "max_tokens": 150,
    },
    "llm": {
        "provider": PROVIDER_OLLAMA,
        "model": "llama3.2:3b",
        "api_base": "http://localhost:11434",
        "api_key": "",
        "fallback_provider": PROVIDER_OLLAMA,
        "fallback_model": "llama3.2:3b",
    },
    "tts": {
        "engine": "tts.piper",
        "voice": "en_US-ryan-medium",
        "speed": 1.0,
        "pitch": 0,
    },
}
