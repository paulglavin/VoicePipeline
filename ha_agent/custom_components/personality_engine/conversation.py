"""PersonalityConversationAgent — HA Conversation platform for Personality Engine.

Flow per utterance:
  1. Extract satellite_id (device_id) and conversation_id from ConversationInput
  2. Look up the most recent speaker from the webhook cache (within 2 s window)
  3. Load per-speaker user config (falls back to "default")
  4. Heuristic: is this likely a local HA intent (lights, locks, etc.)?
  5. If yes: delegate to the built-in "homeassistant" conversation agent
  6. If the intent matched: return the HA response (HA intent scripts handle TTS)
  7. If no match or conversational query: route to per-user LLM
  8. Resolve satellite → media_player and dispatch TTS
  9. Return ConversationResult carrying the LLM response text

Phase 2 extension points (marked with # PHASE2):
  - Conversation history / multi-turn context per speaker
  - HA state injection into system prompt (who's home, device states)
  - Confidence-weighted identity disclosure to LLM
"""

import logging
import time
from typing import Literal

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.intent import IntentResponse, IntentResponseType

from .const import DATA_CONFIG_MANAGER, DATA_SPEAKER_CACHE, DOMAIN
from .llm_router import LLMProviderError, route_to_llm

_LOGGER = logging.getLogger(__name__)

# Seconds back from now to search in the speaker cache.
# Must be > worst-case pipeline latency between webhook and transcript arrival.
_CACHE_WINDOW_SECONDS = 2.0

# Keywords that strongly suggest a local HA control intent
_CONTROL_KEYWORDS: frozenset[str] = frozenset({
    "turn on", "turn off", "switch on", "switch off",
    "set", "dim", "brighten", "open", "close", "lock", "unlock",
    "activate", "deactivate", "enable", "disable",
    "play", "pause", "stop", "next", "previous",
    "increase", "decrease", "raise", "lower",
    "run", "start", "cancel",
})

# Patterns that strongly suggest a conversational / LLM query
_CONVERSATIONAL_PATTERNS: frozenset[str] = frozenset({
    "what", "why", "how", "who", "when", "where",
    "tell me", "explain", "describe", "give me",
    "joke", "story", "weather", "news",
    "i feel", "i think", "i want to talk",
    "remind me", "remember",
    "help me understand",
})


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Personality Engine conversation platform."""
    agent = PersonalityConversationAgent(hass, config_entry)
    async_add_entities([agent])


class PersonalityConversationAgent(ConversationEntity):
    """Speaker-aware personality conversation agent.

    Registered as a ConversationEntity so HA's Assist pipeline can select it
    as the conversation agent for voice commands.
    """

    _attr_name = "Personality Engine"
    _attr_unique_id = "personality_engine_agent"
    _attr_has_entity_name = True

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        self.hass = hass
        self._config_entry = config_entry

    # ------------------------------------------------------------------
    # ConversationEntity interface

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages. Phase 2: read from config."""
        return ["en"]

    @property
    def _config_manager(self):
        return self.hass.data[DOMAIN][DATA_CONFIG_MANAGER]

    # ------------------------------------------------------------------

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process one utterance through the personality pipeline.

        Flow:
          1. Identify speaker from webhook cache
          2. Try local HA intent so the action actually executes (lights turn on, etc.)
          3. Always route through LLM for the spoken response — HA's generic
             "Turned on the light" is replaced with a personality-flavoured reply.
             The ha_action_result context tells the LLM what was done so it can
             confirm it in character.
          4. On LLM failure: fall back to the HA response if one exists.

        Args:
            user_input: ConversationInput from the HA Assist pipeline.

        Returns:
            ConversationResult with the LLM-generated response text.
        """
        _LOGGER.warning(
            "PersonalityEngine v2: async_process called text=%r", user_input.text[:60]
        )
        conversation_id: str | None = user_input.conversation_id
        device_id: str | None = getattr(user_input, "device_id", None)

        # ── 1. Speaker lookup ──────────────────────────────────────────
        cache_entry = self._get_recent_speaker_from_cache()
        if cache_entry:
            speaker_id     = cache_entry["speaker_id"]
            confidence     = cache_entry["confidence"]
        else:
            speaker_id     = "default"
            confidence     = 0.0

        _LOGGER.info(
            "PersonalityEngine: conv=%s device=%s speaker=%s conf=%.2f text=%r",
            conversation_id,
            device_id,
            speaker_id,
            confidence,
            user_input.text[:60],
        )

        # ── 2. User config ─────────────────────────────────────────────
        user_config  = self._config_manager.get_user_config(speaker_id)
        display_name = user_config.get("display_name", speaker_id.capitalize())

        # ── 3. Try local intent (executes the HA action) ───────────────
        ha_action_result: str | None = None
        if self._is_likely_local_intent(user_input.text):
            local_result = await self._try_local_intent(user_input)
            if local_result is not None:
                ha_action_result = (
                    local_result.response.speech
                    .get("plain", {})
                    .get("speech", "")
                )
                _LOGGER.debug(
                    "PersonalityEngine: local intent matched — HA says %r", ha_action_result
                )

        # ── 4. LLM response (always) ───────────────────────────────────
        try:
            llm_response = await self._generate_llm_response(
                user_input.text, user_config, display_name, ha_action_result
            )
        except LLMProviderError as exc:
            _LOGGER.error("PersonalityEngine: LLM failed for %s: %s", speaker_id, exc)
            llm_response = ha_action_result or (
                "Sorry, I'm having trouble thinking right now. Please try again."
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception(
                "PersonalityEngine: unexpected error for %s: %s", speaker_id, exc
            )
            llm_response = ha_action_result or (
                "Sorry, I'm having trouble thinking right now. Please try again."
            )

        # ── 5. Return result — HA pipeline handles TTS ─────────────────
        response = IntentResponse(language=user_input.language)
        response.async_set_speech(llm_response)

        return ConversationResult(
            response=response,
            conversation_id=conversation_id,
        )

    # ------------------------------------------------------------------
    # Speaker cache

    def _get_recent_speaker_from_cache(self) -> dict | None:
        """Return the most recent speaker cache entry within the cache window.

        Returns:
            Cache entry dict, or None if no recent entry exists.
        """
        cache: dict = self.hass.data.get(DOMAIN, {}).get(DATA_SPEAKER_CACHE, {})
        if not cache:
            return None

        now = time.time()
        cutoff = now - _CACHE_WINDOW_SECONDS
        recent = [
            entry for entry in cache.values()
            if entry["timestamp"] >= cutoff
        ]

        if not recent:
            return None

        return max(recent, key=lambda e: e["timestamp"])

    # ------------------------------------------------------------------
    # Intent detection

    def _is_likely_local_intent(self, text: str) -> bool:
        """Heuristic: determine whether text looks like a HA control intent.

        Errs on the side of trying local intent — if HA can't match it,
        we fall through to the LLM anyway.

        Args:
            text: Utterance text (lowercase comparison applied internally).

        Returns:
            True if the text should be tried as a local intent first.
        """
        lower = text.lower()

        # Conversational patterns are a strong signal to skip local intent
        for pattern in _CONVERSATIONAL_PATTERNS:
            if pattern in lower:
                _LOGGER.debug("Intent heuristic: conversational pattern %r", pattern)
                return False

        # Control keywords suggest a local intent
        for keyword in _CONTROL_KEYWORDS:
            if keyword in lower:
                _LOGGER.debug("Intent heuristic: control keyword %r", keyword)
                return True

        # Default: try local intent (conservative — avoids unnecessary LLM calls
        # for short commands like "lights off")
        return True

    async def _try_local_intent(
        self, user_input: ConversationInput
    ) -> ConversationResult | None:
        """Delegate to the built-in HA conversation agent.

        Args:
            user_input: Original ConversationInput.

        Returns:
            ConversationResult if the intent was recognised, None otherwise.
        """
        try:
            result = await conversation.async_converse(
                hass=self.hass,
                text=user_input.text,
                conversation_id=user_input.conversation_id,
                context=user_input.context,
                language=user_input.language,
                agent_id="conversation.home_assistant",
                device_id=getattr(user_input, "device_id", None),
            )
        except Exception as exc:
            _LOGGER.debug("Local intent attempt raised: %s", exc)
            return None

        response_type = result.response.response_type
        matched = response_type in (
            IntentResponseType.ACTION_DONE,
            IntentResponseType.QUERY_ANSWER,
            IntentResponseType.PARTIAL_ACTION_DONE,
        )

        if matched:
            _LOGGER.debug("Local intent matched (type=%s)", response_type)
            return result

        _LOGGER.debug("Local intent not matched (type=%s)", response_type)
        return None

    # ------------------------------------------------------------------
    # LLM

    async def _generate_llm_response(
        self,
        user_message: str,
        user_config: dict,
        display_name: str,
        ha_action_result: str | None = None,
    ) -> str:
        """Generate a personalised LLM response.

        Args:
            user_message:     The user's utterance text.
            user_config:      Full per-user config dict.
            display_name:     Human-readable name for the speaker.
            ha_action_result: HA's confirmation text if a local intent was
                              executed (e.g. "Turned on the light").  When set,
                              the LLM is told the action already happened and
                              should confirm it in character rather than attempt
                              to perform it again.

        Returns:
            Generated response text.

        Raises:
            LLMProviderError: If all LLM providers fail.
        """
        personality = user_config.get("personality", {})
        llm_cfg     = user_config.get("llm", {})

        # PHASE2: inject HA state context (who's home, active devices, time of day)
        system_prompt = personality.get(
            "system_prompt",
            "You are a helpful home assistant. Keep responses concise.",
        )
        system_prompt = f"The user's name is {display_name}. {system_prompt}"

        if ha_action_result:
            system_prompt += (
                f"\n\nThe home automation system has already executed the user's request. "
                f"HA confirmed: '{ha_action_result}'. "
                f"Briefly confirm what was done, in character. One sentence."
            )

        return await route_to_llm(
            hass=self.hass,
            provider=llm_cfg.get("provider", "ollama"),
            model=llm_cfg.get("model", "llama3.2:3b"),
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=personality.get("temperature", 0.7),
            max_tokens=personality.get("max_tokens", 150),
            config=llm_cfg,
            fallback_provider=llm_cfg.get("fallback_provider"),
            fallback_model=llm_cfg.get("fallback_model"),
        )

    # ------------------------------------------------------------------
    # Helpers

    async def _resolve_satellite_entity(self, device_id: str | None) -> str | None:
        """Resolve a HA device_id to the satellite's entity ID.

        ConversationInput.device_id is a HA device registry ID, not an entity
        ID. We look up the assist_satellite entity for this device.

        Args:
            device_id: HA device registry ID, or None.

        Returns:
            Satellite entity ID string, or None.
        """
        if not device_id:
            return None

        from homeassistant.helpers import entity_registry as er

        ent_reg = er.async_get(self.hass)
        device_entities = er.async_entries_for_device(ent_reg, device_id)

        for ent in device_entities:
            if ent.domain == "assist_satellite":
                return ent.entity_id

        # Fallback: return the first entity for this device (best effort)
        if device_entities:
            _LOGGER.debug(
                "No assist_satellite entity for device %s — using %s as proxy",
                device_id,
                device_entities[0].entity_id,
            )
            return device_entities[0].entity_id

        return None
