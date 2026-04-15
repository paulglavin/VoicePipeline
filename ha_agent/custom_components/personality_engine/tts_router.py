"""TTS routing for the Personality Engine.

Resolves a satellite entity ID to a media_player and calls the tts.speak
service to deliver the response to the correct device.

satellite_id resolution strategy (in order):
  1. Direct name substitution: assist_satellite.kitchen → media_player.kitchen
  2. Satellite entity attributes (media_player_entity_id if set)
  3. Area-based lookup: find a media_player in the same area as the satellite

Phase 2 extension point: add per-user overrides so a speaker can always
target a preferred media_player regardless of which satellite heard them.
"""

import logging

from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)

_LOGGER = logging.getLogger(__name__)

# Engines that support the "speed" TTS option
_SPEED_CAPABLE_ENGINES = {"tts.google_cloud", "tts.nabu_casa_cloud", "tts.piper"}

# Engines that support the "pitch" TTS option
_PITCH_CAPABLE_ENGINES = {"tts.google_cloud"}


async def resolve_satellite_media_player(
    hass: HomeAssistant,
    satellite_id: str | None,
) -> str | None:
    """Resolve a satellite entity ID to its associated media_player entity ID.

    Args:
        hass:         HomeAssistant instance.
        satellite_id: Entity ID of the originating satellite
                      (e.g. "assist_satellite.kitchen_assist_satellite"),
                      or None if unknown.

    Returns:
        media_player entity ID string, or None if resolution fails.
    """
    if not satellite_id:
        _LOGGER.warning("TTS router: satellite_id is None — cannot resolve media_player")
        return None

    # ── Strategy 1: direct name substitution ──────────────────────────────
    candidate = (
        satellite_id
        .replace("assist_satellite.", "media_player.")
        .replace("_assist_satellite", "")
    )
    if hass.states.get(candidate) is not None:
        _LOGGER.debug("TTS router: resolved %s → %s (direct)", satellite_id, candidate)
        return candidate

    # ── Strategy 2: entity attribute media_player_entity_id ───────────────
    state = hass.states.get(satellite_id)
    if state is not None:
        attr_mp = state.attributes.get("media_player_entity_id")
        if attr_mp and hass.states.get(attr_mp) is not None:
            _LOGGER.debug(
                "TTS router: resolved %s → %s (attribute)", satellite_id, attr_mp
            )
            return attr_mp

    # ── Strategy 3: area-based lookup ─────────────────────────────────────
    ent_reg = er.async_get(hass)
    dev_reg = dr.async_get(hass)
    area_reg = ar.async_get(hass)

    sat_entry = ent_reg.async_get(satellite_id)
    if sat_entry is None:
        _LOGGER.warning(
            "TTS router: satellite %r not in entity registry", satellite_id
        )
        return None

    # Prefer the device's area, fall back to entity's area
    area_id: str | None = None
    if sat_entry.device_id:
        device = dev_reg.async_get(sat_entry.device_id)
        if device:
            area_id = device.area_id

    if not area_id:
        area_id = sat_entry.area_id

    if not area_id:
        _LOGGER.warning(
            "TTS router: satellite %r has no area — cannot resolve media_player",
            satellite_id,
        )
        return None

    # Find a media_player in the same device first, then same area
    if sat_entry.device_id:
        device_entities = er.async_entries_for_device(ent_reg, sat_entry.device_id)
        for ent in device_entities:
            if ent.domain == "media_player" and hass.states.get(ent.entity_id):
                _LOGGER.debug(
                    "TTS router: resolved %s → %s (same device)",
                    satellite_id,
                    ent.entity_id,
                )
                return ent.entity_id

    area_entities = er.async_entries_for_area(ent_reg, area_id)
    for ent in area_entities:
        if ent.domain == "media_player" and hass.states.get(ent.entity_id):
            _LOGGER.debug(
                "TTS router: resolved %s → %s (same area %r)",
                satellite_id,
                ent.entity_id,
                area_reg.async_get_area(area_id).name if area_reg.async_get_area(area_id) else area_id,
            )
            return ent.entity_id

    _LOGGER.error(
        "TTS router: no media_player found for satellite %r (area=%r)",
        satellite_id,
        area_id,
    )
    return None


async def synthesize_speech(
    hass: HomeAssistant,
    message: str,
    tts_config: dict,
    target_device: str | None,
) -> None:
    """Dispatch a TTS speak service call to the target media_player.

    Silently no-ops (with a warning) if target_device is None so that LLM
    responses are never silently dropped without a log entry.

    Args:
        hass:          HomeAssistant instance.
        message:       The text to speak.
        tts_config:    TTS configuration dict from user config
                       (engine, voice, speed, pitch).
        target_device: media_player entity ID to speak on.
    """
    if not target_device:
        _LOGGER.warning(
            "TTS router: no target_device — dropping response: %r", message[:80]
        )
        return

    engine: str = tts_config.get("engine", "tts.piper")

    service_data: dict = {
        "entity_id": target_device,
        "message": message,
        "cache": False,
    }

    # Build engine-specific options
    options: dict = {}
    if "voice" in tts_config and tts_config["voice"]:
        options["voice"] = tts_config["voice"]
    if engine in _SPEED_CAPABLE_ENGINES:
        options["speed"] = float(tts_config.get("speed", 1.0))
    if engine in _PITCH_CAPABLE_ENGINES:
        options["pitch"] = int(tts_config.get("pitch", 0))

    if options:
        service_data["options"] = options

    _LOGGER.debug(
        "TTS router: %s → %s | engine=%s voice=%s",
        message[:40],
        target_device,
        engine,
        options.get("voice", "default"),
    )

    try:
        await hass.services.async_call(
            "tts",
            "speak",
            service_data,
            blocking=False,
        )
    except Exception as exc:
        _LOGGER.error(
            "TTS router: tts.speak failed for %s: %s", target_device, exc
        )
