"""Claude through the Anthropic Messages API, directly.

The ReAct loop speaks one message format (the chat-completions shape) and is
never told which vendor is on the other end. This client translates at the
boundary, and it is where every difference that would otherwise surface as a
wrong number, rather than an error, is handled:

* **No sampling parameters.** Current Claude models reject `temperature`
  outright (a 400), so it is sent only to the older models that still accept
  it. Everywhere else the model samples at its own default, and the game log
  says so -- a run's variance then includes the model's own sampling, which is
  a limitation to report, not noise to average away.
* **Thinking is always on** (Claude Opus 5.5) and counts toward `max_tokens`.
  The harness's reply cap was sized for models that do not think; left as is,
  it would cut the model off mid-thought and every turn would fall through to
  the malformed-reply path. The cap here has room for the thinking too, and
  `effort` -- recorded in the log -- is what bounds how much it thinks.
* **Thinking is part of the history.** A tool-use loop must hand each thinking
  block back exactly as it came, so the reply's own content blocks travel with
  the assistant message (`LLMResponse.native_content`) instead of being rebuilt
  from the parsed call. And a block is only valid in the conversation that
  produced it: editing anything before it is a 400 on new accounts. The loop
  therefore does not trim a Claude turn (`append_only`), and
  `prefix_mismatch_behavior: drop_block` turns any edit that slips through into
  a counted drop instead of a crashed turn.
* **One call per step.** The loop acts on one tool call per step; a second,
  parallel call would be left without a result, which the API rejects. Parallel
  tool use is switched off, and any call that still arrives unanswered is given
  an explicit "not executed" result rather than silently dropped.
* **Refusals are data.** A safety decline is an HTTP 200 with
  `stop_reason: "refusal"`. It is surfaced as its own outcome, never as a
  malformed reply -- in an injection study, what the model declined is part of
  what is being measured.
* **Fallbacks are attributed.** With fallbacks on, a declined request can be
  answered by another model. That turn is then tagged with the model that
  actually answered it, because a seat silently changing model would corrupt
  the one thing a per-seat log exists to record.

The `anthropic` SDK is imported lazily, so the offline path and the test suite
run without it installed.
"""

from __future__ import annotations

import json
import time

from .base import LLMClient, LLMResponse, ProviderError, ToolCall

DEFAULT_BASE_URL = "https://api.anthropic.com"

# Models that still take `temperature`. Everything newer rejects it with a 400,
# so the default is to leave it out -- omitting it is never an error.
_SAMPLING_PREFIXES = (
    "claude-haiku-4-5",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-1",
    "claude-opus-4-0",
    "claude-sonnet-4-0",
    "claude-3",
)

# Models whose thinking is configured with `adaptive` + `effort`. Older ones
# (Haiku 4.5 and before) take neither, and run here without thinking.
_ADAPTIVE_PREFIXES = (
    "claude-fable",
    "claude-mythos",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
)

# Server-side refusal fallbacks exist only for these, on the Claude API.
_FALLBACK_PREFIXES = ("claude-fable-5-1", "claude-opus-5", "claude-sonnet-5-5")

BETA_FALLBACK = "server-side-fallback-2026-07-01"
BETA_BLOCK_BINDING = "thinking-binding-controls-2026-08-01"

NOT_EXECUTED = (
    "Not executed: only one action is carried out per step. "
    "Call it again on its own if you still need it."
)


class AnthropicClient(LLMClient):
    tool_mode = "native"
    # The loop may not rewrite earlier messages of a turn for this client:
    # see the module docstring.
    append_only = True

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        display_name: str | None = None,
        effort: str = "medium",
        thinking_display: str = "summarized",
        max_tokens: int = 16000,
        min_timeout: float = 120.0,
        fallbacks: str | None = "default",
        prompt_cache: bool = True,
        max_sdk_retries: int = 2,
    ):
        if not api_key:
            raise ProviderError(
                "no Anthropic API key configured",
                hint="Export ANTHROPIC_API_KEY, or add an Anthropic provider on "
                     "the config page.",
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover -- exercised by hand
            raise ProviderError(
                "the anthropic package is not installed",
                hint="pip install anthropic",
            ) from exc

        self._sdk = anthropic
        self.model = model
        self.name = display_name or model
        self.api_key = api_key
        # The SDK appends /v1 itself. A base URL copied with /v1 on the end
        # would become /v1/v1/messages -- a 404 that reads like a bad model name.
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.group = None
        self.effort = effort
        self.thinking_display = thinking_display
        self.max_tokens = max_tokens
        self.min_timeout = min_timeout
        self.prompt_cache = prompt_cache
        self.adaptive = model.startswith(_ADAPTIVE_PREFIXES)
        self.accepts_sampling = model.startswith(_SAMPLING_PREFIXES)
        self.fallbacks = fallbacks if model.startswith(_FALLBACK_PREFIXES) else None
        # The SDK's own retries honour `retry-after` on a 429, which the loop's
        # fixed backoff does not. A rate limit that exhausted the loop's retries
        # would make the seat abstain -- changing the game over what is an
        # infrastructure problem -- so both layers stay on.
        self._client = anthropic.Anthropic(
            api_key=api_key,
            base_url=self.base_url,
            max_retries=max_sdk_retries,
        )

    def describe(self) -> dict:
        """What this client actually sent, for the game log. Every setting that
        can change a model's behaviour is a variable of the experiment."""
        return {
            "provider": "anthropic",
            "model": self.model,
            "effort": self.effort if self.adaptive else None,
            "thinking": (
                f"adaptive ({self.thinking_display})" if self.adaptive else "off"
            ),
            "sampling": "temperature" if self.accepts_sampling else "model default",
            "max_tokens": self.max_tokens,
            "context_policy": "append_only",
            "fallbacks": self.fallbacks or "off",
            "prompt_cache": self.prompt_cache,
        }

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 800,
        timeout: float = 30.0,
    ) -> LLMResponse:
        system, wire = to_anthropic(messages)
        betas = [BETA_BLOCK_BINDING] if self.adaptive else []
        kwargs: dict = {
            "model": self.model,
            # The loop's cap is sized for a reply alone; thinking needs room on
            # top of it. The larger of the two always wins.
            "max_tokens": max(max_tokens, self.max_tokens),
            "messages": wire,
            "timeout": max(timeout, self.min_timeout),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [to_anthropic_tool(t) for t in tools]
            kwargs["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        if self.adaptive:
            kwargs["thinking"] = {
                "type": "adaptive",
                "display": self.thinking_display,
                "block_binding": {"prefix_mismatch_behavior": "drop_block"},
            }
            kwargs["output_config"] = {"effort": self.effort}
        if self.accepts_sampling:
            kwargs["extra_body"] = {"temperature": temperature}
        if self.prompt_cache:
            kwargs["cache_control"] = {"type": "ephemeral"}
        if self.fallbacks:
            kwargs["fallbacks"] = self.fallbacks
            betas.append(BETA_FALLBACK)
        if betas:
            kwargs["betas"] = betas

        started = time.time()
        try:
            message = self._client.beta.messages.create(**kwargs)
        except self._sdk.APIStatusError as exc:
            raise ProviderError(
                f"HTTP {exc.status_code} from Anthropic: {_error_text(exc)}",
                hint=explain_anthropic(exc.status_code, _error_text(exc), self.model),
                status=exc.status_code,
            ) from exc
        except self._sdk.APITimeoutError as exc:
            raise ProviderError(
                f"timed out after {kwargs['timeout']}s",
                hint="The model is thinking for longer than the timeout allows; "
                     "lower effort or raise the timeout.",
            ) from exc
        except self._sdk.APIConnectionError as exc:
            raise ProviderError(
                f"cannot reach {self.base_url}: {exc}",
                hint="Check the network. From a sandbox, api.anthropic.com must be "
                     "on the allowlist.",
            ) from exc
        latency_ms = int((time.time() - started) * 1000)
        return self._to_response(message, latency_ms)

    def _to_response(self, message, latency_ms: int) -> LLMResponse:
        usage = message.usage
        cached_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cached_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        # `input_tokens` excludes cached tokens. The harness reports raw token
        # counts so that runs are comparable whatever the cache did; billed
        # cost is a separate question.
        prompt_tokens = (usage.input_tokens or 0) + cached_read + cached_write

        iterations = getattr(usage, "iterations", None) or []
        fallback_ran = any(getattr(i, "type", "") == "fallback_message" for i in iterations)

        response = LLMResponse(
            prompt_tokens=prompt_tokens,
            completion_tokens=usage.output_tokens or 0,
            latency_ms=latency_ms,
            finish_reason=message.stop_reason or "end_turn",
            model=message.model or self.model,
        )
        response.cache_read_tokens = cached_read

        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            response.refusal = getattr(details, "category", None) or "unspecified"
            return response  # nothing to act on, and nothing to hand back

        if fallback_ran:
            response.served_by = message.model

        texts, thinking = [], []
        for block in message.content:
            kind = getattr(block, "type", "")
            if kind == "text" and block.text:
                texts.append(block.text)
            elif kind == "thinking" and getattr(block, "thinking", ""):
                thinking.append(block.thinking)
            elif kind == "tool_use":
                args = block.input
                response.tool_calls.append(
                    ToolCall(
                        name=block.name,
                        arguments=args if isinstance(args, dict) else {},
                        id=block.id,
                        raw_arguments=json.dumps(args, ensure_ascii=False),
                        malformed=not isinstance(args, dict),
                    )
                )
        response.text = "\n".join(texts).strip()
        response.thinking = "\n\n".join(thinking).strip()

        # Hand the reply back verbatim -- thinking blocks, signatures and all --
        # but only if it holds something the API will accept as a turn. A reply
        # cut off mid-thought has neither text nor a call to replay.
        if any(getattr(b, "type", "") in ("text", "tool_use") for b in message.content):
            response.native_content = list(message.content)
        return response

    def list_models(self, timeout: float = 15.0) -> list[str]:
        try:
            return [m.id for m in self._client.models.list(timeout=timeout)]
        except self._sdk.APIError as exc:
            raise ProviderError(f"cannot list models: {exc}") from exc


# ------------------------------------------------------------ translation

def to_anthropic_tool(schema: dict) -> dict:
    fn = schema.get("function", schema)
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """Chat-completions history -> (system, Messages API history).

    Deterministic by construction: the same history always translates to the
    same bytes, which is what keeps an append-only history append-only on the
    wire, and keeps the prompt cache warm.
    """
    system_parts: list[str] = []
    out: list[dict] = []

    def user_turn() -> dict:
        if not out or out[-1]["role"] != "user":
            out.append({"role": "user", "content": []})
        return out[-1]

    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            if content:
                system_parts.append(content)
        elif role == "assistant":
            native = m.get("native_content")
            if native:
                blocks = list(native)
            else:
                blocks = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for call in m.get("tool_calls") or []:
                    fn = call.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    blocks.append({"type": "tool_use", "id": call.get("id", "call_0"),
                                   "name": fn.get("name", ""),
                                   "input": args if isinstance(args, dict) else {}})
            if blocks:  # an empty assistant turn is a 400; drop it
                out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            user_turn()["content"].append({
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", "call_0"),
                "content": content or "",
            })
        elif content:  # user
            user_turn()["content"].append({"type": "text", "text": content})

    _answer_every_call(out)
    for turn in out:
        if turn["role"] == "user":
            # Results first, then text: the API wants a tool result directly
            # after the call it answers.
            turn["content"].sort(key=lambda b: b.get("type") != "tool_result")
    return "\n\n".join(system_parts), out


def _answer_every_call(out: list[dict]) -> None:
    """Every tool call needs a result in the very next message, or the request
    is rejected. The loop answers one call per step, so anything else gets an
    explicit refusal to run it."""
    for i, turn in enumerate(out):
        if turn["role"] != "assistant":
            continue
        ids = [_field(b, "id") for b in turn["content"] if _field(b, "type") == "tool_use"]
        if not ids:
            continue
        if i + 1 == len(out):
            continue  # the call is the last thing said; its result is not due yet
        nxt = out[i + 1]
        if nxt["role"] != "user":
            nxt = {"role": "user", "content": []}
            out.insert(i + 1, nxt)
        answered = {b.get("tool_use_id") for b in nxt["content"]
                    if b.get("type") == "tool_result"}
        for tid in ids:
            if tid not in answered:
                nxt["content"].append({"type": "tool_result", "tool_use_id": tid,
                                       "content": NOT_EXECUTED, "is_error": True})


def _field(block, name: str):
    return block.get(name) if isinstance(block, dict) else getattr(block, name, None)


# ------------------------------------------------------------------ errors

def _error_text(exc) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") or {}
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:600]
    return str(exc)[:600]


def explain_anthropic(status: int | None, detail: str, model: str) -> str:
    low = (detail or "").lower()
    if status == 401:
        return "The API key was rejected. Re-copy it in full (it starts with sk-ant-)."
    if status == 403:
        return "The key is valid but lacks permission for this model or workspace."
    if status == 404:
        return (f"Unknown model {model!r}. Use an exact ID such as claude-opus-5-5 "
                "or claude-sonnet-5-5 -- no date suffix.")
    if status == 429:
        return "Rate limited. Lower the runner's worker count."
    if status == 529:
        return "Anthropic is temporarily overloaded; the loop retries this."
    if status == 400:
        if "temperature" in low or "top_p" in low:
            return "This model takes no sampling parameters; the client should not send them."
        if "credit" in low or "billing" in low:
            return "The account is out of credit."
        if "thinking" in low:
            return "The thinking configuration is not valid for this model."
        return "The request was rejected as invalid; the raw message is in the details."
    if status and status >= 500:
        return "Anthropic's API failed; the loop retries this."
    return "Unrecognised API error; the raw response is in the details field."
