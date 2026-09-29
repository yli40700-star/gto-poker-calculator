"""Gateway probe.

Phase-0 checklist, run before any experiment code is trusted. Every item here
is a failure that would otherwise surface halfway through a paid batch:

  1. basic chat works at all
  2. a request carrying `tools` comes back with a real tool_calls structure
  3. the arguments parse as JSON
  4. a tool result can be fed back and the conversation continues
  5. temperature=0 is actually honoured (some upstreams ignore it)
  6. the model name is known to the gateway
  7. round-trip latency, for setting timeouts
  8. token accounting is reported at all (needed for the cost axis)

A model that fails (2)-(4) is not unusable -- it is demoted to
`tool_mode="json_prompt"`, and the probe records which mode it qualified for so
the game log can carry it.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from .base import ProviderError
from .openai_compat import OpenAICompatClient

# A relay hiccup must not be reported as "this model does not work". Retries
# live in the ReAct loop's recovery path, which the probe does not go through,
# so it needs its own -- a single transient 503 during phase 0 would otherwise
# demote a perfectly good model and quietly change which models enter the
# experiment.
RETRYABLE = {408, 409, 429, 500, 502, 503, 504, 529}
ATTEMPTS = 3
BACKOFF_S = 1.5


def _with_retry(call, notes: list[str]):
    """Run a probe request, retrying only what is worth retrying."""
    delay = BACKOFF_S
    for attempt in range(ATTEMPTS):
        try:
            return call()
        except ProviderError as exc:
            transient = exc.status in RETRYABLE or "timed out" in str(exc).lower()
            if not transient or attempt == ATTEMPTS - 1:
                raise
            notes.append(f"transient {exc.status or 'timeout'} on attempt "
                         f"{attempt + 1}, retrying")
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")  # pragma: no cover


PROBE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "report_number",
            "description": "Report a single number back to the caller.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "integer", "description": "the number"}
                },
                "required": ["value"],
            },
        },
    }
]


@dataclass
class ProbeResult:
    model: str
    reachable: bool = False
    native_tools: bool = False
    args_parse: bool = False
    multi_turn_tools: bool = False
    temperature_stable: bool | None = None
    reports_usage: bool = False
    latency_ms: int = 0
    tool_mode: str = "json_prompt"
    error: str | None = None
    hint: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def probe_model(
    client,
    *,
    check_temperature: bool = True,
    timeout: float = 30.0,
) -> ProbeResult:
    result = ProbeResult(model=client.model)

    # 1. reachability + usage accounting
    started = time.time()
    try:
        basic = _with_retry(lambda: client.chat(
            [{"role": "user", "content": "Reply with the single word: ready"}],
            temperature=0.0,
            max_tokens=16,
            timeout=timeout,
        ), result.notes)
    except ProviderError as exc:
        result.error = str(exc)[:400]
        result.hint = exc.hint
        return result
    result.reachable = True
    result.latency_ms = int((time.time() - started) * 1000)
    result.reports_usage = basic.prompt_tokens > 0 or basic.completion_tokens > 0
    if not result.reports_usage:
        result.notes.append(
            "gateway returned no usage field; token cost must be estimated locally"
        )

    # 2-3. native tool calling and argument parsing. A relay model is tried in
    # native mode whatever it is configured for; Claude only has one mode.
    native_client = client
    if isinstance(client, OpenAICompatClient):
        native_client = OpenAICompatClient(
            model=client.model,
            api_key=client.api_key,
            base_url=client.base_url,
            tool_mode="native",
            group=client.group,
        )
    try:
        tooled = _with_retry(lambda: native_client.chat(
            [{"role": "user", "content": "Call report_number with value 7."}],
            tools=PROBE_TOOL,
            temperature=0.0,
            max_tokens=64,
            timeout=timeout,
        ), result.notes)
        calls = [c for c in tooled.tool_calls if c.name == "report_number"]
        result.native_tools = bool(calls)
        result.args_parse = bool(calls) and not calls[0].malformed
    except ProviderError as exc:
        result.notes.append(f"tool request failed: {str(exc)[:160]}")

    # 4. feeding a tool result back
    if result.native_tools:
        try:
            call = tooled.tool_calls[0]
            follow = native_client.chat(
                [
                    {"role": "user", "content": "Call report_number with value 7."},
                    {
                        "role": "assistant",
                        "content": None,
                        # Replayed verbatim where the provider returned its own
                        # blocks -- a thinking model needs its thinking back.
                        "native_content": tooled.native_content,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.raw_arguments or "{}",
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": call.id, "content": "ok"},
                    {"role": "user", "content": "Now say 'done' in one word."},
                ],
                tools=PROBE_TOOL,
                temperature=0.0,
                max_tokens=32,
                timeout=timeout,
            )
            result.multi_turn_tools = bool(follow.text or follow.tool_calls)
        except ProviderError as exc:
            result.notes.append(f"tool result round-trip failed: {str(exc)[:160]}")

    result.tool_mode = (
        "native" if (result.native_tools and result.args_parse and result.multi_turn_tools)
        else "json_prompt"
    )
    if not isinstance(client, OpenAICompatClient) and client.tool_mode != "native":
        # A client that only speaks the JSON protocol (claude -p): the checks
        # above ran through it, so they test the protocol, not native calling.
        result.tool_mode = client.tool_mode
        result.notes.append("tool calls go through the JSON protocol")
    if result.tool_mode == "json_prompt" and isinstance(client, OpenAICompatClient):
        result.notes.append(
            "no usable native function calling; this model runs in JSON-prompt mode"
        )

    # 5. is temperature honoured? Current Claude models take no sampling
    # parameters at all, so there is nothing to check -- and nothing to claim.
    if check_temperature and getattr(client, "accepts_sampling", True) is False:
        result.notes.append(
            "this model takes no sampling parameters; it samples at its own "
            "default, so repeated runs are not deterministic (report as a limitation)"
        )
    elif check_temperature:
        try:
            outs = {
                client.chat(
                    [{"role": "user", "content": "Name one colour. One word only."}],
                    temperature=0.0,
                    max_tokens=8,
                    timeout=timeout,
                ).text.strip().lower()
                for _ in range(3)
            }
            result.temperature_stable = len(outs) == 1
            if not result.temperature_stable:
                result.notes.append(
                    "temperature=0 did not produce identical output three times; "
                    "the upstream may ignore the parameter (note it as a limitation)"
                )
        except ProviderError as exc:
            result.notes.append(f"temperature check failed: {str(exc)[:160]}")

    return result
