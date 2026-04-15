"""Personality Engine — speaker-aware personality agent for HA Assist.

Integration entry points:
  async_setup           — component-level setup (no-op, entry-based)
  async_setup_entry     — initialise storage, webhook, speaker cache, agent
  async_unload_entry    — clean up

Webhook (POST /api/webhook/personality_engine_input):
  VoicePipeline POSTs speaker metadata here immediately after speaker
  identification, before the Wyoming transcript arrives at HA Assist.
  The webhook stores the result in an in-memory cache keyed by timestamp
  so the conversation agent can correlate it with the incoming utterance.

Race safety:
  Multiple satellites may fire concurrently. Each webhook entry receives a
  unique timestamped key so entries never overwrite each other. The agent
  reads the most recent entry within a 2-second window.
"""

import logging
import re
import time

from aiohttp import web
from homeassistant.components.webhook import (
    async_register as webhook_async_register,
    async_unregister as webhook_async_unregister,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import (
    DATA_CONFIG_MANAGER,
    DATA_SPEAKER_CACHE,
    DATA_USER_MAP,
    DOMAIN,
    SPEAKER_CACHE_TTL,
    WEBHOOK_ID,
)
from .user_config import UserConfigManager

_LOGGER = logging.getLogger(__name__)

# Valid speaker_id: alphanumeric + underscore, 1–50 characters
_SPEAKER_ID_RE = re.compile(r"^[a-z0-9_]{1,50}$")


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Component-level setup (YAML config not used — entry-based only)."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Personality Engine from a config entry.

    Initialises:
      - UserConfigManager backed by HA persistent storage
      - In-memory speaker metadata cache
      - Speaker → HA user_id mapping
      - Webhook for VoicePipeline speaker notifications
      - PersonalityConversationAgent (via conversation platform)

    Args:
        hass:  HomeAssistant instance.
        entry: Config entry created by the config flow.

    Returns:
        True on success.
    """
    hass.data.setdefault(DOMAIN, {})

    # ── User config manager ────────────────────────────────────────────
    config_manager = UserConfigManager(hass)
    await config_manager.async_load()

    hass.data[DOMAIN][DATA_CONFIG_MANAGER] = config_manager
    hass.data[DOMAIN][DATA_SPEAKER_CACHE]  = {}
    hass.data[DOMAIN][DATA_USER_MAP]       = {}

    # ── Webhook registration ───────────────────────────────────────────
    webhook_async_register(
        hass,
        DOMAIN,
        "Personality Engine Input",
        WEBHOOK_ID,
        _async_handle_webhook,
    )
    _LOGGER.info(
        "Personality Engine: webhook registered at /api/webhook/%s", WEBHOOK_ID
    )

    # ── Conversation platform (registers PersonalityConversationAgent) ──
    await hass.config_entries.async_forward_entry_setups(entry, ["conversation"])

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry and clean up resources.

    Args:
        hass:  HomeAssistant instance.
        entry: Config entry being unloaded.

    Returns:
        True if unload succeeded.
    """
    unloaded = await hass.config_entries.async_unload_platforms(entry, ["conversation"])
    if unloaded:
        webhook_async_unregister(hass, WEBHOOK_ID)
        hass.data.pop(DOMAIN, None)
        _LOGGER.info("Personality Engine: unloaded")

    return unloaded


# ---------------------------------------------------------------------------
# Webhook handler


async def _async_handle_webhook(
    hass: HomeAssistant,
    webhook_id: str,
    request: web.Request,
) -> web.Response:
    """Handle a POST from VoicePipeline with speaker identification metadata.

    Expected payload::

        {
            "speaker_id":     "paul",
            "confidence":     0.92,
            "timestamp":      "2026-04-15T12:34:56.789Z",
            "interaction_id": "550e8400-e29b-41d4-a716-446655440000"  // optional
        }

    Args:
        hass:       HomeAssistant instance.
        webhook_id: Registered webhook ID (always WEBHOOK_ID).
        request:    Incoming aiohttp request.

    Returns:
        JSON response: {"success": true} or {"success": false, "error": "..."}.
    """
    try:
        data = await request.json()
    except Exception:
        _LOGGER.warning("Personality Engine webhook: invalid JSON body")
        return web.json_response({"success": False, "error": "invalid JSON"}, status=400)

    # ── Validate ───────────────────────────────────────────────────────
    speaker_id = str(data.get("speaker_id", "")).strip().lower()
    if not speaker_id or not _SPEAKER_ID_RE.match(speaker_id):
        _LOGGER.warning(
            "Personality Engine webhook: invalid speaker_id %r", speaker_id
        )
        return web.json_response(
            {"success": False, "error": "invalid speaker_id"}, status=422
        )

    raw_confidence = data.get("confidence")
    if raw_confidence is None:
        _LOGGER.warning("Personality Engine webhook: missing confidence field")
        return web.json_response(
            {"success": False, "error": "missing confidence"}, status=422
        )

    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        _LOGGER.warning(
            "Personality Engine webhook: non-numeric confidence %r", raw_confidence
        )
        return web.json_response(
            {"success": False, "error": "confidence must be numeric"}, status=422
        )

    interaction_id: str | None = data.get("interaction_id")

    # ── Map speaker_id to HA user_id ───────────────────────────────────
    user_id = await _get_or_create_user_id(hass, speaker_id)

    # ── Store in cache (unique timestamped key for race safety) ────────
    cache_key = f"speaker_{int(time.time() * 1000)}_{speaker_id}"
    hass.data[DOMAIN][DATA_SPEAKER_CACHE][cache_key] = {
        "user_id":        user_id,
        "speaker_id":     speaker_id,
        "confidence":     confidence,
        "interaction_id": interaction_id,
        "timestamp":      time.time(),
    }

    _LOGGER.debug(
        "Personality Engine webhook: cached %s conf=%.2f key=%s",
        speaker_id, confidence, cache_key,
    )

    # ── Purge stale entries ────────────────────────────────────────────
    _cleanup_speaker_cache(hass)

    return web.json_response({"success": True})


async def _get_or_create_user_id(hass: HomeAssistant, speaker_id: str) -> str:
    """Return the HA user_id for *speaker_id*, creating a shadow user if needed.

    Shadow users are system-generated (not visible in HA user management) and
    exist solely to provide a stable user_id for per-speaker context.
    Phase 2: link shadow users to real HA accounts via config flow.

    Args:
        hass:       HomeAssistant instance.
        speaker_id: Speaker identifier string.

    Returns:
        HA user_id string.
    """
    user_map: dict[str, str] = hass.data[DOMAIN][DATA_USER_MAP]

    # Return cached mapping first (avoids auth lookup on every utterance)
    if speaker_id in user_map:
        return user_map[speaker_id]

    # Check for existing shadow user by name
    display_name = f"voice_speaker_{speaker_id}"
    try:
        users = await hass.auth.async_get_users()
        for user in users:
            if user.name == display_name and user.system_generated:
                user_map[speaker_id] = user.id
                _LOGGER.debug(
                    "Personality Engine: found existing shadow user %s → %s",
                    speaker_id, user.id,
                )
                return user.id

        # Create new shadow user
        new_user = await hass.auth.async_create_user(
            name=display_name,
            group_ids=[],
            local_only=True,
        )
        user_map[speaker_id] = new_user.id
        _LOGGER.info(
            "Personality Engine: created shadow user %s → %s",
            speaker_id, new_user.id,
        )
        return new_user.id

    except Exception as exc:
        # Auth failure is non-fatal — fall back to speaker_id as surrogate
        _LOGGER.warning(
            "Personality Engine: shadow user lookup/create failed for %s: %s",
            speaker_id, exc,
        )
        return speaker_id


def _cleanup_speaker_cache(hass: HomeAssistant) -> None:
    """Remove speaker cache entries older than SPEAKER_CACHE_TTL seconds.

    Args:
        hass: HomeAssistant instance.
    """
    cache: dict = hass.data.get(DOMAIN, {}).get(DATA_SPEAKER_CACHE, {})
    cutoff = time.time() - SPEAKER_CACHE_TTL
    stale_keys = [k for k, v in cache.items() if v["timestamp"] < cutoff]
    for key in stale_keys:
        del cache[key]

    if stale_keys:
        _LOGGER.debug("Personality Engine: purged %d stale cache entries", len(stale_keys))
