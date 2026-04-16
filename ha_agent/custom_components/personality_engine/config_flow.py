"""Config flow for the Personality Engine integration.

Initial setup: one-step confirmation (no credentials required at install time).
Per-user configuration: handled entirely through the Options flow, accessible
from the integration page in Settings → Devices & Services.

Options flow menu:
  Add User      — create a new speaker personality
  Edit User     — modify an existing speaker's config
  Delete User   — remove a speaker (default user is protected)
"""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    DEFAULT_USER_CONFIG,
    DOMAIN,
    PROVIDER_ANTHROPIC,
    PROVIDER_LLAMACPP,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    DATA_CONFIG_MANAGER,
)

_LOGGER = logging.getLogger(__name__)

_LLM_PROVIDER_OPTIONS = [
    selector.SelectOptionDict(value=PROVIDER_OLLAMA,    label="Ollama (local)"),
    selector.SelectOptionDict(value=PROVIDER_ANTHROPIC, label="Anthropic (cloud)"),
    selector.SelectOptionDict(value=PROVIDER_OPENAI,    label="OpenAI (cloud)"),
    selector.SelectOptionDict(value=PROVIDER_LLAMACPP,  label="llama.cpp (local)"),
]


class PersonalityEngineConfigFlow(
    config_entries.ConfigFlow, domain=DOMAIN
):
    """Handle the initial integration setup."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Show a confirmation form and create the config entry on submit.

        Args:
            user_input: Form data on submit, None on first render.

        Returns:
            FlowResult (form or create_entry).
        """
        # Only one instance allowed
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()

        if user_input is not None:
            return self.async_create_entry(
                title="Personality Engine",
                data={},
            )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({}),
            description_placeholders={},
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> PersonalityEngineOptionsFlow:
        """Return the options flow handler."""
        return PersonalityEngineOptionsFlow()


class PersonalityEngineOptionsFlow(config_entries.OptionsFlow):
    """Handle per-user personality configuration via the options flow."""

    def __init__(self) -> None:
        self._speaker_id: str | None = None
        self._existing_config: dict = {}

    # ------------------------------------------------------------------
    # Menu

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Show the top-level options menu."""
        return self.async_show_menu(
            step_id="init",
            menu_options=["add_user", "edit_user", "delete_user"],
        )

    # ------------------------------------------------------------------
    # Add user

    async def async_step_add_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Collect speaker_id and display_name for a new user.

        Args:
            user_input: Form data, or None on first render.

        Returns:
            FlowResult.
        """
        errors: dict[str, str] = {}

        if user_input is not None:
            speaker_id   = user_input["speaker_id"].strip().lower()
            display_name = user_input["display_name"].strip()

            if not speaker_id.replace("_", "").isalnum():
                errors["speaker_id"] = "invalid_speaker_id"
            elif len(speaker_id) > 50:
                errors["speaker_id"] = "speaker_id_too_long"
            else:
                self._speaker_id = speaker_id
                self._existing_config = {
                    "display_name": display_name,
                    **{k: v for k, v in DEFAULT_USER_CONFIG.items() if k != "display_name"},
                }
                return await self.async_step_configure_user()

        return self.async_show_form(
            step_id="add_user",
            data_schema=vol.Schema(
                {
                    vol.Required("speaker_id"): str,
                    vol.Required("display_name"): str,
                }
            ),
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Edit user

    async def async_step_edit_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Show a selector of existing users to edit.

        Args:
            user_input: Form data, or None on first render.

        Returns:
            FlowResult.
        """
        config_manager = self.hass.data[DOMAIN][DATA_CONFIG_MANAGER]
        users = config_manager.list_users()

        if not users:
            return self.async_abort(reason="no_users")

        if user_input is not None:
            speaker_id = user_input["speaker_id"]
            self._speaker_id = speaker_id
            self._existing_config = config_manager.get_user_config(speaker_id)
            return await self.async_step_configure_user()

        return self.async_show_form(
            step_id="edit_user",
            data_schema=vol.Schema(
                {
                    vol.Required("speaker_id"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                selector.SelectOptionDict(value=u, label=u)
                                for u in users
                            ]
                        )
                    )
                }
            ),
        )

    # ------------------------------------------------------------------
    # Configure user (shared by add and edit)

    async def async_step_configure_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Form for full personality / LLM / TTS configuration.

        Args:
            user_input: Form data, or None on first render.

        Returns:
            FlowResult.
        """
        existing = self._existing_config
        p = existing.get("personality", DEFAULT_USER_CONFIG["personality"])
        llm = existing.get("llm", DEFAULT_USER_CONFIG["llm"])
        tts = existing.get("tts", DEFAULT_USER_CONFIG["tts"])

        if user_input is not None:
            config = {
                "display_name": user_input["display_name"],
                "personality": {
                    "system_prompt": user_input["system_prompt"],
                    "temperature":   float(user_input.get("temperature", 0.7)),
                    "max_tokens":    int(user_input.get("max_tokens", 150)),
                },
                "llm": {
                    "provider":          user_input["llm_provider"],
                    "model":             user_input["llm_model"],
                    "api_base":          user_input.get("llm_api_base", ""),
                    "api_key":           user_input.get("llm_api_key", ""),
                    "fallback_provider": user_input.get("fallback_provider", ""),
                    "fallback_model":    user_input.get("fallback_model", ""),
                },
                "tts": {
                    "engine": user_input.get("tts_engine", "tts.piper"),
                    "voice":  user_input.get("tts_voice", ""),
                    "speed":  float(user_input.get("tts_speed", 1.0)),
                    "pitch":  int(user_input.get("tts_pitch", 0)),
                },
            }

            config_manager = self.hass.data[DOMAIN][DATA_CONFIG_MANAGER]
            await config_manager.async_add_user(self._speaker_id, config)

            return self.async_create_entry(title="", data={})

        schema = vol.Schema(
            {
                vol.Required(
                    "display_name", default=existing.get("display_name", "")
                ): str,
                vol.Required(
                    "system_prompt",
                    default=p.get("system_prompt", DEFAULT_USER_CONFIG["personality"]["system_prompt"]),
                ): selector.TextSelector(
                    selector.TextSelectorConfig(multiline=True)
                ),
                vol.Required(
                    "llm_provider", default=llm.get("provider", PROVIDER_OLLAMA)
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=_LLM_PROVIDER_OPTIONS)
                ),
                vol.Required(
                    "llm_model", default=llm.get("model", "llama3.2:3b")
                ): str,
                vol.Optional(
                    "llm_api_base", default=llm.get("api_base", "")
                ): str,
                vol.Optional(
                    "llm_api_key", default=""
                ): selector.TextSelector(
                    selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
                ),
                vol.Optional(
                    "fallback_provider", default=llm.get("fallback_provider", "")
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[selector.SelectOptionDict(value="", label="None")]
                        + _LLM_PROVIDER_OPTIONS,
                    )
                ),
                vol.Optional(
                    "fallback_model", default=llm.get("fallback_model", "")
                ): str,
                vol.Optional(
                    "tts_engine", default=tts.get("engine", "tts.piper")
                ): str,
                vol.Optional(
                    "tts_voice", default=tts.get("voice", "")
                ): str,
                vol.Optional(
                    "tts_speed", default=tts.get("speed", 1.0)
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0.5, max=2.0, step=0.1)
                ),
                vol.Optional(
                    "tts_pitch", default=tts.get("pitch", 0)
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=-20, max=20, step=1)
                ),
            }
        )

        return self.async_show_form(
            step_id="configure_user",
            data_schema=schema,
            description_placeholders={"name": existing.get("display_name", "the speaker")},
        )

    # ------------------------------------------------------------------
    # Delete user

    async def async_step_delete_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Show a selector of deletable users (excludes "default").

        Args:
            user_input: Form data, or None on first render.

        Returns:
            FlowResult.
        """
        config_manager = self.hass.data[DOMAIN][DATA_CONFIG_MANAGER]
        users = [u for u in config_manager.list_users() if u != "default"]

        if not users:
            return self.async_abort(reason="no_deletable_users")

        if user_input is not None:
            speaker_id = user_input["speaker_id"]
            await config_manager.async_delete_user(speaker_id)
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="delete_user",
            data_schema=vol.Schema(
                {
                    vol.Required("speaker_id"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                selector.SelectOptionDict(value=u, label=u)
                                for u in users
                            ]
                        )
                    )
                }
            ),
        )
