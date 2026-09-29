"""Model access.

Three ways to reach a model: an OpenAI-compatible gateway (any vendor, through
a relay), Claude through the Anthropic API (billed per token), or Claude
through `claude -p` (the Claude Code CLI, on a Claude subscription). Which one is decided by
the *provider*, never by the model name -- a relay serves Claude models too,
under the same names.
"""

from __future__ import annotations

import os

from .base import (
    LLMClient,
    LLMResponse,
    ProviderError,
    ProviderExhausted,
    ToolCall,
    parse_json_action,
)
from .anthropic_client import AnthropicClient
from .claude_cli import ClaudeCLIClient
from .mock import MockClient
from .openai_compat import OpenAICompatClient, explain
from .probe import ProbeResult, probe_model

DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")


def build_client(config: dict) -> LLMClient:
    """Build a client from a model config dict.

    `{"model_name": "mock"}` yields the offline scripted client, so the whole
    pipeline runs with no key and no network -- that is what the tests use.
    """
    model = config.get("model_name") or config.get("model") or "mock"
    if model == "mock":
        return MockClient(
            seed=int(config.get("seed", 0)),
            susceptibility=float(config.get("susceptibility", 0.75)),
        )
    if provider_kind(config) == "claude_cli":
        return ClaudeCLIClient(
            model=model,
            display_name=config.get("display_name"),
            effort=config.get("effort") or "medium",
            executable=config.get("executable"),
        )
    if provider_kind(config) == "anthropic":
        return AnthropicClient(
            model=model,
            api_key=config.get("api_key") or os.getenv("ANTHROPIC_API_KEY", ""),
            base_url=config.get("base_url") or None,
            display_name=config.get("display_name"),
            effort=config.get("effort") or "medium",
            fallbacks=None if config.get("fallbacks") == "off" else "default",
        )
    api_key = config.get("api_key") or os.getenv("LLM_API_KEY", "")
    return OpenAICompatClient(
        model=model,
        api_key=api_key,
        base_url=config.get("base_url") or DEFAULT_BASE_URL,
        tool_mode=config.get("tool_mode", "native"),
        display_name=config.get("display_name"),
        group=config.get("group"),
    )


def provider_kind(config: dict) -> str:
    """"claude_cli", "anthropic" or "openai_compat". Explicit wins; otherwise the base URL
    decides. The model name never does: a relay serves claude-* names too."""
    kind = config.get("provider") or config.get("provider_kind")
    if kind:
        return kind
    return "anthropic" if "api.anthropic.com" in (config.get("base_url") or "") else "openai_compat"


__all__ = [
    "AnthropicClient",
    "ClaudeCLIClient",
    "ProviderExhausted",
    "DEFAULT_BASE_URL",
    "LLMClient",
    "LLMResponse",
    "MockClient",
    "OpenAICompatClient",
    "ProbeResult",
    "ProviderError",
    "ToolCall",
    "build_client",
    "explain",
    "parse_json_action",
    "probe_model",
    "provider_kind",
]
