"""LLM provider abstraction and routing for the Personality Engine.

Supported providers:
  - ollama      — local Ollama server (recommended for privacy/latency)
  - anthropic   — Anthropic cloud API (claude-sonnet-4-6 recommended)
  - openai      — OpenAI cloud API
  - llamacpp    — llama.cpp server (OpenAI-compatible API)

All providers implement the LLMProvider ABC with a single async generate()
method. The route_to_llm() function handles primary + optional fallback logic.

Phase 2 extension point: add new providers by subclassing LLMProvider and
registering in LLM_REGISTRY.
"""

import asyncio
import logging
from abc import ABC, abstractmethod

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    PROVIDER_ANTHROPIC,
    PROVIDER_LLAMACPP,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
)

_LOGGER = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Raised when an LLM provider call fails after all retries."""


class LLMProvider(ABC):
    """Abstract base class for LLM provider implementations."""

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        config: dict,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            system_prompt: System/persona instructions.
            user_message:  The user's utterance text.
            temperature:   Sampling temperature (0.0–1.0).
            max_tokens:    Maximum tokens in the response.
            config:        Full LLM config dict for this user (api_base, api_key, etc.).

        Returns:
            The generated response string.

        Raises:
            LLMProviderError: If the request fails.
        """


# ---------------------------------------------------------------------------
# Ollama


class OllamaProvider(LLMProvider):
    """Ollama local LLM server (http://localhost:11434 by default).

    Recommended model: llama3.2:3b — fast, good instruction following, <2 GB RAM.
    """

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        config: dict,
    ) -> str:
        """Generate via Ollama /api/chat endpoint."""
        api_base = config.get("api_base", "http://localhost:11434").rstrip("/")
        model = config.get("model", "llama3.2:3b")
        url = f"{api_base}/api/chat"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        session = async_get_clientsession(self.hass)
        try:
            async with asyncio.timeout(10):
                resp = await session.post(url, json=payload)
                resp.raise_for_status()
                data = await resp.json()
        except asyncio.TimeoutError as exc:
            raise LLMProviderError(f"Ollama request timed out after 10 s") from exc
        except Exception as exc:
            raise LLMProviderError(f"Ollama request failed: {exc}") from exc

        try:
            return data["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise LLMProviderError(f"Unexpected Ollama response shape: {data}") from exc


# ---------------------------------------------------------------------------
# Anthropic


class AnthropicProvider(LLMProvider):
    """Anthropic cloud API (api.anthropic.com).

    Recommended model: claude-sonnet-4-6 — best reasoning, ~500–1200 ms latency.
    Requires api_key in user config.
    """

    _API_URL = "https://api.anthropic.com/v1/messages"
    _API_VERSION = "2023-06-01"

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        config: dict,
    ) -> str:
        """Generate via Anthropic Messages API."""
        api_key = config.get("api_key", "")
        if not api_key:
            raise LLMProviderError("Anthropic api_key is not configured")

        model = config.get("model", "claude-sonnet-4-6")

        headers = {
            "x-api-key": api_key,
            "anthropic-version": self._API_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }

        session = async_get_clientsession(self.hass)
        try:
            async with asyncio.timeout(15):
                resp = await session.post(self._API_URL, json=payload, headers=headers)
                resp.raise_for_status()
                data = await resp.json()
        except asyncio.TimeoutError as exc:
            raise LLMProviderError("Anthropic request timed out after 15 s") from exc
        except Exception as exc:
            raise LLMProviderError(f"Anthropic request failed: {exc}") from exc

        try:
            return data["content"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(f"Unexpected Anthropic response shape: {data}") from exc


# ---------------------------------------------------------------------------
# OpenAI


class OpenAIProvider(LLMProvider):
    """OpenAI cloud API (api.openai.com).

    Also compatible with Azure OpenAI when api_base is overridden.
    """

    _DEFAULT_API_BASE = "https://api.openai.com"

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        config: dict,
    ) -> str:
        """Generate via OpenAI Chat Completions API."""
        api_key = config.get("api_key", "")
        if not api_key:
            raise LLMProviderError("OpenAI api_key is not configured")

        api_base = config.get("api_base", self._DEFAULT_API_BASE).rstrip("/")
        model = config.get("model", "gpt-4o-mini")
        url = f"{api_base}/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        session = async_get_clientsession(self.hass)
        try:
            async with asyncio.timeout(15):
                resp = await session.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = await resp.json()
        except asyncio.TimeoutError as exc:
            raise LLMProviderError("OpenAI request timed out after 15 s") from exc
        except Exception as exc:
            raise LLMProviderError(f"OpenAI request failed: {exc}") from exc

        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(f"Unexpected OpenAI response shape: {data}") from exc


# ---------------------------------------------------------------------------
# llama.cpp


class LlamaCppProvider(LLMProvider):
    """llama.cpp server (OpenAI-compatible /v1/chat/completions endpoint).

    Run llama.cpp with: ./server -m model.gguf --port 8080
    Set api_base to http://<host>:8080 in user config.
    """

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        config: dict,
    ) -> str:
        """Generate via llama.cpp OpenAI-compatible API."""
        api_base = config.get("api_base", "http://localhost:8080").rstrip("/")
        model = config.get("model", "local-model")
        url = f"{api_base}/v1/chat/completions"

        # llama.cpp doesn't require Authorization but accepts it harmlessly
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = config.get("api_key", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        session = async_get_clientsession(self.hass)
        try:
            async with asyncio.timeout(15):
                resp = await session.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = await resp.json()
        except asyncio.TimeoutError as exc:
            raise LLMProviderError("llama.cpp request timed out after 15 s") from exc
        except Exception as exc:
            raise LLMProviderError(f"llama.cpp request failed: {exc}") from exc

        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(f"Unexpected llama.cpp response shape: {data}") from exc


# ---------------------------------------------------------------------------
# Provider registry

LLM_REGISTRY: dict[str, type[LLMProvider]] = {
    PROVIDER_OLLAMA:    OllamaProvider,
    PROVIDER_ANTHROPIC: AnthropicProvider,
    PROVIDER_OPENAI:    OpenAIProvider,
    PROVIDER_LLAMACPP:  LlamaCppProvider,
}


# ---------------------------------------------------------------------------
# Router


async def route_to_llm(
    hass: HomeAssistant,
    provider: str,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
    config: dict,
    fallback_provider: str | None = None,
    fallback_model: str | None = None,
) -> str:
    """Route an LLM request to the configured provider with optional fallback.

    Tries the primary provider first. If it fails and a fallback is configured,
    attempts the fallback. Raises LLMProviderError if both fail.

    Args:
        hass:              HomeAssistant instance.
        provider:          Primary provider identifier (e.g. "ollama").
        model:             Primary model name.
        system_prompt:     System/persona instructions.
        user_message:      The user's utterance text.
        temperature:       Sampling temperature.
        max_tokens:        Maximum response tokens.
        config:            Full LLM config dict for the user.
        fallback_provider: Optional fallback provider identifier.
        fallback_model:    Optional fallback model name.

    Returns:
        Generated response text.

    Raises:
        LLMProviderError: If all attempts fail.
    """
    primary_cls = LLM_REGISTRY.get(provider)
    if primary_cls is None:
        raise LLMProviderError(f"Unknown LLM provider: {provider!r}")

    # Build a config copy with the correct model for this attempt
    primary_config = {**config, "model": model}

    _LOGGER.debug("LLM: trying %s / %s", provider, model)
    try:
        response = await primary_cls(hass).generate(
            system_prompt, user_message, temperature, max_tokens, primary_config
        )
        _LOGGER.debug("LLM: %s succeeded (%d chars)", provider, len(response))
        return response
    except LLMProviderError as primary_exc:
        _LOGGER.warning("LLM primary %s/%s failed: %s", provider, model, primary_exc)

    if fallback_provider and fallback_model:
        fallback_cls = LLM_REGISTRY.get(fallback_provider)
        if fallback_cls is None:
            raise LLMProviderError(
                f"Primary {provider}/{model} failed and fallback provider "
                f"{fallback_provider!r} is unknown"
            )

        fallback_config = {**config, "model": fallback_model}
        _LOGGER.info("LLM: falling back to %s / %s", fallback_provider, fallback_model)
        try:
            response = await fallback_cls(hass).generate(
                system_prompt, user_message, temperature, max_tokens, fallback_config
            )
            _LOGGER.debug("LLM: fallback %s succeeded (%d chars)", fallback_provider, len(response))
            return response
        except LLMProviderError as fallback_exc:
            _LOGGER.error(
                "LLM fallback %s/%s also failed: %s", fallback_provider, fallback_model, fallback_exc
            )
            raise LLMProviderError(
                f"All LLM providers failed. Primary: {primary_exc}. Fallback: {fallback_exc}"
            ) from fallback_exc

    raise LLMProviderError(f"LLM provider {provider}/{model} failed and no fallback configured")
