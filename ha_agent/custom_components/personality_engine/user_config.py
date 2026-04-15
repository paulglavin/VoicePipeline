"""User configuration manager for the Personality Engine.

Stores per-speaker personality, LLM, and TTS configuration in HA's encrypted
persistent storage (.storage/personality_engine_users).
"""

import copy
import logging
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DEFAULT_USER_CONFIG, STORAGE_KEY, STORAGE_VERSION

_LOGGER = logging.getLogger(__name__)


class UserConfigManager:
    """Manages per-speaker personality configuration backed by HA storage.

    All public async methods are safe to call from the event loop.
    Thread safety: HA storage helpers are not thread-safe; call only from
    the event loop thread (never via asyncio.to_thread / run_in_executor).
    """

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self._store: Store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self._config: dict | None = None

    # ------------------------------------------------------------------
    # Load / save

    async def async_load(self) -> dict:
        """Load configuration from storage, seeding defaults if absent.

        Returns:
            The full configuration dict (version + users).
        """
        if self._config is not None:
            return self._config

        stored = await self._store.async_load()
        if stored is None:
            self._config = self._get_default_config()
            await self._store.async_save(self._config)
            _LOGGER.debug("Personality Engine: seeded default user config")
        else:
            self._config = stored

        return self._config

    async def async_save(self, config: dict) -> None:
        """Persist configuration to storage and update the in-memory cache.

        Args:
            config: Full configuration dict to persist.
        """
        self._config = config
        await self._store.async_save(config)

    # ------------------------------------------------------------------
    # Read

    def get_user_config(self, speaker_id: str) -> dict:
        """Return the configuration for *speaker_id*, falling back to "default".

        Merges the speaker config on top of the default so keys added in
        future versions are always present.

        Args:
            speaker_id: Lowercase speaker identifier (e.g. "paul").

        Returns:
            A deep copy of the merged configuration dict.
        """
        if self._config is None:
            _LOGGER.warning(
                "get_user_config called before async_load — returning default"
            )
            return copy.deepcopy(DEFAULT_USER_CONFIG)

        users = self._config.get("users", {})
        base = copy.deepcopy(DEFAULT_USER_CONFIG)

        # Merge default user over baseline defaults
        if "default" in users:
            _deep_merge(base, users["default"])

        # Merge speaker-specific config on top (if different from default)
        if speaker_id in users and speaker_id != "default":
            _deep_merge(base, users[speaker_id])

        return base

    def list_users(self) -> list[str]:
        """Return sorted list of all configured speaker IDs.

        Returns:
            Sorted list of speaker_id strings.
        """
        if self._config is None:
            return []
        return sorted(self._config.get("users", {}).keys())

    # ------------------------------------------------------------------
    # Write

    async def async_add_user(self, speaker_id: str, config: dict) -> None:
        """Add or update a speaker's configuration and persist immediately.

        Args:
            speaker_id: Speaker identifier to add or update.
            config: Configuration dict (will be merged into existing if present).
        """
        await self.async_load()
        self._config["users"][speaker_id] = config  # type: ignore[index]
        await self.async_save(self._config)  # type: ignore[arg-type]
        _LOGGER.info("Personality Engine: saved config for speaker %r", speaker_id)

    async def async_delete_user(self, speaker_id: str) -> None:
        """Remove a speaker's configuration.

        The "default" speaker cannot be deleted.

        Args:
            speaker_id: Speaker identifier to remove.

        Raises:
            ValueError: If speaker_id is "default".
        """
        if speaker_id == "default":
            raise ValueError("Cannot delete the 'default' user")

        await self.async_load()
        users = self._config.get("users", {})  # type: ignore[union-attr]
        if speaker_id in users:
            del users[speaker_id]
            await self.async_save(self._config)  # type: ignore[arg-type]
            _LOGGER.info("Personality Engine: deleted config for speaker %r", speaker_id)
        else:
            _LOGGER.debug("async_delete_user: %r not found, no-op", speaker_id)

    # ------------------------------------------------------------------
    # Defaults

    def _get_default_config(self) -> dict:
        """Return the factory-default configuration structure.

        Returns:
            Dict with version and a single "default" user entry.
        """
        return {
            "version": STORAGE_VERSION,
            "users": {
                "default": copy.deepcopy(DEFAULT_USER_CONFIG),
            },
        }


# ---------------------------------------------------------------------------
# Helpers


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in-place.

    Dict values are merged; all other types are replaced.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
