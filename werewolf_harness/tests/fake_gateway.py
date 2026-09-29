"""A local stand-in for an OpenAI-compatible gateway.

Not a mock of the *model* -- that is `providers/mock.py`. This is a mock of the
*wire*: a real HTTP server that speaks the chat-completions API, so the real
`OpenAICompatClient` can be driven end to end without a network or a key.

It exists because the offline client bypasses every part of the stack that the
first paid run depends on: the HTTP layer, native `tools` / `tool_calls`
round-tripping, the assistant/tool message pairing, usage accounting, retries
and the error mapping. Those paths only ever get exercised against a server, and
"it worked against the relay" is an expensive way to find out they don't.

    python -m werewolf_harness.tests.fake_gateway          # serve on :8900
    python -m werewolf_harness.tests.fake_gateway --check   # run games through it

It speaks two protocols on the same port: chat-completions (the relay path)
and the Anthropic Messages API (`/v1/messages`, the direct-Claude path, driven
through the real `anthropic` SDK). The Messages side enforces the rules the
real API enforces and *records* every violation -- a 400 is swallowed by the
loop's recovery and turns into an abstention, so "the game finished" proves
nothing unless the server also says nothing was rejected.

The server plays by delegating to the offline client, so a game against it is a
real game over real HTTP -- and just as scripted, and just as excluded from any
reported result.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..harness.providers.mock import MockClient

MODELS = ["fake-native-4o", "fake-json-only", "fake-flaky"]

# Anthropic side. Suffixes select a behaviour; the prefix is what the client
# keys its request shape on, so these are treated exactly as the real IDs are.
CLAUDE_MODELS = {
    "claude-opus-5-5": "normal",
    "claude-opus-5-5-refuses": "refuse",
    "claude-opus-5-5-fallback": "fallback",
    "claude-opus-5-5-flaky": "flaky",
    "claude-haiku-4-5": "normal",
}
_NO_SAMPLING = ("claude-opus-5", "claude-sonnet-5", "claude-fable")
_ADAPTIVE = ("claude-opus-5", "claude-sonnet-5", "claude-fable")


class Ledger:
    """What the Messages endpoint saw. Reset per check."""

    def __init__(self):
        self.requests: list[dict] = []
        self.violations: list[str] = []
        self.issued: dict[str, str] = {}      # signature -> thinking text
        self.tool_thinking: dict[str, str] = {}  # tool_use id -> signature
        self.replayed_ok = 0
        self.dropped = 0
        self.fail_next = 0


LEDGER = Ledger()


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _prefix_hash(body: dict, upto: int) -> str:
    """The conversation a thinking block was produced in: everything before it."""
    return hashlib.sha256(_canon({
        "system": body.get("system"),
        "tools": body.get("tools"),
        "messages": body["messages"][:upto],
    }).encode()).hexdigest()


def _sign(prefix: str, text: str) -> str:
    return hashlib.sha256(f"{prefix}|{text}".encode()).hexdigest()


class Handler(BaseHTTPRequestHandler):
    client = MockClient(seed=0)
    calls: list[dict] = []
    fail_next = 0  # set by the flaky model to exercise the retry path

    def log_message(self, *args):  # keep the test output readable
        pass

    def do_GET(self):
        if self.path.split("?")[0].rstrip("/") == "/v1/models" and self.headers.get("x-api-key"):
            self._json({"data": [{"id": m, "type": "model", "display_name": m,
                                  "created_at": "2026-01-01T00:00:00Z"}
                                 for m in CLAUDE_MODELS],
                        "has_more": False, "first_id": None, "last_id": None})
        elif self.path.rstrip("/").endswith("/models"):
            self._json({"object": "list",
                        "data": [{"id": m, "object": "model"} for m in MODELS]})
        else:
            self._json({"error": {"message": "not found"}}, status=404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.startswith("/v1/messages"):
            return self._messages(body)
        Handler.calls.append(body)

        if not (self.headers.get("Authorization") or "").startswith("Bearer sk-"):
            return self._json({"error": {"message": "Invalid token"}}, status=401)

        model = body.get("model", "")
        if model not in MODELS:
            return self._json(
                {"error": {"message": f"No available channel for model {model}"}},
                status=400,
            )
        if model == "fake-flaky" and Handler.fail_next > 0:
            Handler.fail_next -= 1
            return self._json({"error": {"message": "upstream busy"}}, status=503)

        response = Handler.client.chat(body["messages"], tools=body.get("tools"))
        call = response.tool_calls[0] if response.tool_calls else None

        if model == "fake-json-only":
            # A model with no native tool calling: answer in prose, as one does.
            message = {"role": "assistant",
                       "content": json.dumps({"thought": "…",
                                              "action": call.name if call else "speak",
                                              "args": call.arguments if call else {}})}
        else:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_%d" % len(Handler.calls),
                    "type": "function",
                    "function": {"name": call.name,
                                 "arguments": json.dumps(call.arguments)},
                }] if call else [],
            }

        self._json({
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if call else "stop"}],
            "usage": {"prompt_tokens": response.prompt_tokens,
                      "completion_tokens": response.completion_tokens,
                      "total_tokens": response.total_tokens},
        })

    # ------------------------------------------------ Anthropic Messages API

    def _error(self, status: int, kind: str, message: str):
        if status == 400:
            LEDGER.violations.append(message)
        return self._json({"type": "error", "error": {"type": kind, "message": message}},
                          status=status)

    def _messages(self, body: dict):
        LEDGER.requests.append(body)
        if not (self.headers.get("x-api-key") or "").startswith("sk-ant-"):
            return self._json({"type": "error", "error": {
                "type": "authentication_error", "message": "invalid x-api-key"}}, 401)
        if not self.headers.get("anthropic-version"):
            return self._error(400, "invalid_request_error", "missing anthropic-version")
        model = body.get("model", "")
        behaviour = CLAUDE_MODELS.get(model)
        if behaviour is None:
            return self._json({"type": "error", "error": {
                "type": "not_found_error", "message": f"model: {model}"}}, 404)
        if behaviour == "flaky" and LEDGER.fail_next > 0:
            LEDGER.fail_next -= 1
            return self._json({"type": "error", "error": {
                "type": "overloaded_error", "message": "Overloaded"}}, 529)

        problem = self._validate(body, model)
        if problem:
            return self._error(400, "invalid_request_error", problem)

        if behaviour == "refuse":
            return self._json({
                "id": "msg_refused", "type": "message", "role": "assistant",
                "model": model, "content": [], "stop_reason": "refusal",
                "stop_sequence": None,
                "stop_details": {"type": "refusal", "category": "cyber",
                                 "explanation": "declined by the fake"},
                "usage": {"input_tokens": 10, "output_tokens": 0},
            })

        if any(x["name"] == "report_number" for x in body.get("tools") or []):
            return self._probe_reply(body, model)

        # Decide by delegating to the offline player, fed a chat-format view.
        chat = [{"role": "system", "content": _flat(body.get("system"))}]
        for m in body["messages"]:
            chat.append({"role": m["role"], "content": _flat(m["content"])})
        tools = [{"type": "function", "function": {"name": x["name"]}}
                 for x in body.get("tools") or []]
        decision = Handler.client.chat(chat, tools=tools or None)
        call = decision.tool_calls[0] if decision.tool_calls else None

        content = []
        if model.startswith(_ADAPTIVE):
            prefix = _prefix_hash(body, len(body["messages"]))
            text = (f"Reviewing the table before {call.name if call else 'replying'}: "
                    "who has contradicted themselves, and what the deaths imply.")
            sig = _sign(prefix, text)
            LEDGER.issued[sig] = text
            content.append({"type": "thinking", "thinking": text, "signature": sig})
        served = model
        iterations = None
        if behaviour == "fallback":
            served = "claude-opus-5"
            content.insert(0, {"type": "fallback", "from": {"model": model},
                               "to": {"model": served}})
            iterations = [{"type": "message", "input_tokens": 10, "output_tokens": 0},
                          {"type": "fallback_message", "input_tokens": 10,
                           "output_tokens": 5, "model": served}]
        if call:
            tid = f"toolu_{len(LEDGER.requests):05d}"
            content.append({"type": "tool_use", "id": tid, "name": call.name,
                            "input": call.arguments})
            thinking = [b for b in content if b["type"] == "thinking"]
            if thinking:
                LEDGER.tool_thinking[tid] = thinking[0]["signature"]
        else:
            content.append({"type": "text", "text": decision.text or "..."})
        usage = {"input_tokens": decision.prompt_tokens, "output_tokens":
                 decision.completion_tokens, "cache_creation_input_tokens": 0,
                 "cache_read_input_tokens": 0}
        if iterations:
            usage["iterations"] = iterations
        self._json({
            "id": f"msg_{len(LEDGER.requests)}", "type": "message", "role": "assistant",
            "model": served, "content": content,
            "stop_reason": "tool_use" if call else "end_turn", "stop_sequence": None,
            "usage": usage,
        })

    def _probe_reply(self, body: dict, model: str):
        """The probe's own tool, answered as a function-calling model would:
        call it when asked, then carry on once its result comes back."""
        last = body["messages"][-1]["content"]
        answered = isinstance(last, list) and any(
            b.get("type") == "tool_result" for b in last)
        content = []
        if model.startswith(_ADAPTIVE):
            prefix = _prefix_hash(body, len(body["messages"]))
            sig = _sign(prefix, "Asked to report 7.")
            LEDGER.issued[sig] = "Asked to report 7."
            content.append({"type": "thinking", "thinking": "Asked to report 7.",
                            "signature": sig})
        if answered:
            content.append({"type": "text", "text": "done"})
        else:
            tid = f"toolu_probe_{len(LEDGER.requests)}"
            content.append({"type": "tool_use", "id": tid, "name": "report_number",
                            "input": {"value": 7}})
            if model.startswith(_ADAPTIVE):
                LEDGER.tool_thinking[tid] = sig
        return self._json({
            "id": "msg_probe", "type": "message", "role": "assistant", "model": model,
            "content": content, "stop_reason": "end_turn" if answered else "tool_use",
            "stop_sequence": None, "usage": {"input_tokens": 20, "output_tokens": 5},
        })

    def _validate(self, body: dict, model: str) -> str | None:
        """The rules the real endpoint enforces that this harness could break."""
        if model.startswith(_NO_SAMPLING):
            for p in ("temperature", "top_p", "top_k"):
                if p in body:
                    return f"{p} is not supported for this model"
        choice = (body.get("tool_choice") or {}).get("type")
        if model.startswith(_ADAPTIVE) and choice in ("any", "tool"):
            return 'tool_choice: type "tool" and "any" are not supported for this model.'
        thinking = body.get("thinking") or {}
        if model.startswith("claude-opus-5-5") and thinking.get("type") in ("disabled", "enabled"):
            return f'"thinking.type.{thinking["type"]}" is not supported for this model.'
        if body.get("max_tokens", 0) < 1024 and model.startswith(_ADAPTIVE):
            # Not a 400 in reality -- the reply is just cut off mid-thought.
            # Recorded here because that is the failure it would cause.
            return "max_tokens too small for a thinking model (reply would be truncated)"

        msgs = body.get("messages") or []
        if not msgs or msgs[0].get("role") != "user":
            return "messages: first message must use the user role"
        for i, m in enumerate(msgs):
            if m.get("role") not in ("user", "assistant"):
                return f"messages.{i}.role: unexpected value {m.get('role')!r}"
            blocks = m.get("content")
            if isinstance(blocks, str):
                blocks = [{"type": "text", "text": blocks}]
            if not blocks:
                return f"messages.{i}: all messages must have non-empty content"
            for b in blocks:
                if b.get("type") == "text" and not b.get("text"):
                    return f"messages.{i}: text content blocks must be non-empty"

            if m["role"] == "assistant":
                uses = [b["id"] for b in blocks if b.get("type") == "tool_use"]
                if uses:
                    nxt = msgs[i + 1]["content"] if i + 1 < len(msgs) else None
                    if nxt is not None:
                        if isinstance(nxt, str) or msgs[i + 1]["role"] != "user":
                            return f"messages.{i+1}: tool_use ids {uses} have no tool_result"
                        results = [b.get("tool_use_id") for b in nxt
                                   if b.get("type") == "tool_result"]
                        missing = [u for u in uses if u not in results]
                        if missing:
                            return f"messages.{i+1}: tool_use ids {missing} have no tool_result"
                        seen_other = False
                        for b in nxt:
                            if b.get("type") != "tool_result":
                                seen_other = True
                            elif seen_other:
                                return f"messages.{i+1}: tool_result blocks must come first"
                    for u in uses:
                        if u in LEDGER.tool_thinking and not any(
                                b.get("type") == "thinking" for b in blocks):
                            return (f"messages.{i}: thinking blocks must be passed back "
                                    "with the tool_use they preceded")
                problem = self._check_thinking(body, i, blocks)
                if problem:
                    return problem
            else:
                prev = msgs[i - 1] if i else None
                prev_uses = set()
                if prev and prev["role"] == "assistant" and not isinstance(prev["content"], str):
                    prev_uses = {b["id"] for b in prev["content"] if b.get("type") == "tool_use"}
                for b in blocks:
                    if b.get("type") == "tool_result" and b.get("tool_use_id") not in prev_uses:
                        return (f"messages.{i}: tool_result {b.get('tool_use_id')} does not "
                                "answer a tool_use in the previous message")
        return None

    def _check_thinking(self, body: dict, i: int, blocks: list[dict]) -> str | None:
        """Preserved thinking: a replayed block must be unmodified and must sit in
        the conversation that produced it."""
        binding = ((body.get("thinking") or {}).get("block_binding") or {})
        for b in blocks:
            if b.get("type") != "thinking":
                continue
            sig = b.get("signature", "")
            if LEDGER.issued.get(sig) != b.get("thinking"):
                return f"messages.{i}: thinking block was modified or not issued here"
            if _sign(_prefix_hash(body, i), b["thinking"]) != sig:
                if binding.get("prefix_mismatch_behavior") == "drop_block":
                    LEDGER.dropped += 1
                    continue
                return (f"messages.{i}: thinking block no longer matches the conversation "
                        "that produced it (an earlier message was edited)")
            LEDGER.replayed_ok += 1
        return None

    def _json(self, payload, status=200):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _flat(content) -> str:
    """Anthropic content -> the plain text the offline player reads."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        if b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif b.get("type") == "tool_result":
            c = b.get("content")
            parts.append(c if isinstance(c, str) else _flat(c))
    return "\n".join(parts)


def serve(port: int = 8900) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _check(port: int = 8900) -> int:
    """Run a full game through the real HTTP client and report what it proved."""
    from ..evalkit.runner import RunConfig, run_game

    server = serve(port)
    base = f"http://127.0.0.1:{port}/v1"
    failures = []

    for model, mode in (("fake-native-4o", "native"), ("fake-json-only", "json_prompt")):
        Handler.calls.clear()
        log = run_game(RunConfig(
            seed=5,
            model={"model_name": model, "display_name": model, "tool_mode": mode,
                   "api_key": "sk-local-fake", "base_url": base},
            guard_layers=("L1", "L2", "L3"),
            attack_enabled=True,
        ))
        turns = sum(len(r["agents"]) + len(r.get("night_turns", []))
                    for r in log["rounds"])
        print(f"{model} ({mode}): crashed={log['outcome']['crashed']} "
              f"winner={log['outcome']['winner']} turns={turns} "
              f"http_calls={len(Handler.calls)} "
              f"tokens={log['outcome']['total_prompt_tokens']}")
        if log["outcome"]["crashed"]:
            failures.append(f"{model}: {log['outcome']['crash_reason']}")

        # The pairing rule real gateways enforce.
        for body in Handler.calls:
            for i, m in enumerate(body["messages"]):
                if m.get("role") == "tool":
                    prev = body["messages"][i - 1]
                    if prev.get("role") != "assistant" or not prev.get("tool_calls"):
                        failures.append(f"{model}: orphaned tool message at {i}")
                    elif m["tool_call_id"] not in {c["id"] for c in prev["tool_calls"]}:
                        failures.append(f"{model}: tool_call_id does not match")
        sent_tools = any("tools" in b for b in Handler.calls)
        if mode == "native" and not sent_tools:
            failures.append("native mode sent no tools field")
        if mode == "json_prompt" and sent_tools:
            failures.append("json mode sent a tools field")

    # Retry path, and the two error mappings that matter most.
    from ..harness.providers import OpenAICompatClient, ProviderError, probe_model

    Handler.fail_next = 2
    client = OpenAICompatClient(model="fake-flaky", api_key="sk-local-fake", base_url=base)
    result = probe_model(client, check_temperature=False)
    print(f"probe(fake-flaky after 2x503): reachable={result.reachable} "
          f"tool_mode={result.tool_mode}")

    for model, key, expect in (("no-such-model", "sk-x", "channel group"),
                               ("fake-native-4o", "bad", "token")):
        try:
            OpenAICompatClient(model=model, api_key=key, base_url=base).chat(
                [{"role": "user", "content": "hi"}])
            failures.append(f"{model}/{key}: no error raised")
        except ProviderError as exc:
            print(f"error mapping [{model} + {key}]: {exc.hint}")
            if expect not in (exc.hint or ""):
                failures.append(f"{model}: hint did not mention {expect!r}")

    server.shutdown()
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nthe real HTTP path is sound: both tool modes, message pairing, "
          "usage accounting, retries and error mapping")
    return 0


def _check_claude(port: int = 8901) -> int:
    """Full games through the real `anthropic` SDK, against rules the real
    Messages API enforces. Each assertion is a failure that would otherwise
    show up as a quietly wrong number in the first paid batch."""
    from ..evalkit.runner import RunConfig, run_game
    from ..harness.providers import AnthropicClient, ProviderError, build_client, probe_model

    server = serve(port)
    base = f"http://127.0.0.1:{port}"
    failures: list[str] = []

    def spec(model: str) -> dict:
        return {"provider": "anthropic", "model_name": model, "display_name": model,
                "api_key": "sk-ant-local-fake", "base_url": base}

    def steps(log):
        for rnd in log["rounds"]:
            for turn in rnd["agents"] + rnd.get("night_turns", []):
                yield turn, turn["react_trace"]

    for guards in ((), ("L1", "L2", "L3")):
        LEDGER.__init__()
        log = run_game(RunConfig(seed=5, model=spec("claude-opus-5-5"),
                                 guard_layers=guards, attack_enabled=True))
        longest = max(len(trace) for _, trace in steps(log))
        thought = sum(1 for _, tr in steps(log) for s in tr if s["thought"].startswith("Reviewing"))
        label = "+".join(guards) or "none"
        print(f"claude-opus-5-5 guard={label}: crashed={log['outcome']['crashed']} "
              f"requests={len(LEDGER.requests)} rejected={len(LEDGER.violations)} "
              f"thinking replayed={LEDGER.replayed_ok} dropped={LEDGER.dropped} "
              f"longest turn={longest} steps")
        if log["outcome"]["crashed"]:
            failures.append(f"{label}: crashed: {log['outcome']['crash_reason']}")
        for v in LEDGER.violations[:3]:
            failures.append(f"{label}: API would reject: {v}")
        if LEDGER.dropped:
            failures.append(f"{label}: history was edited; {LEDGER.dropped} thinking blocks dropped")
        if not LEDGER.replayed_ok:
            failures.append(f"{label}: no thinking block was ever handed back")
        if not thought:
            failures.append(f"{label}: the model's thinking never reached the replay log")
        if any("temperature" in r for r in LEDGER.requests):
            failures.append(f"{label}: temperature sent to a model that rejects it")
        if any(r.get("tools") and not r["tool_choice"].get("disable_parallel_tool_use")
               for r in LEDGER.requests):
            failures.append(f"{label}: parallel tool use left on")
        settings = log["config"].get("provider_settings", {}).get("claude-opus-5-5", {})
        if settings.get("sampling") != "model default" or settings.get("effort") != "medium":
            failures.append(f"{label}: provider settings not recorded: {settings}")

    # Negative control: put the old trimming back. If the check above is worth
    # anything, it must catch this. The window is narrowed for the control only:
    # trimming runs after information lookups, and the offline player makes
    # two per turn at most, where a real model makes several and reaches the
    # default window by itself.
    from ..harness.agent.context import ContextBuilder

    LEDGER.__init__()
    trim = ContextBuilder.trim_steps

    def tight(self, messages):
        saved, self.max_steps_in_context = self.max_steps_in_context, 2
        try:
            return trim(self, messages)
        finally:
            self.max_steps_in_context = saved

    AnthropicClient.append_only = False
    ContextBuilder.trim_steps = tight
    try:
        run_game(RunConfig(seed=5, model=spec("claude-opus-5-5"),
                           guard_layers=("L1", "L2", "L3"), attack_enabled=True))
    finally:
        AnthropicClient.append_only = True
        ContextBuilder.trim_steps = trim
    print(f"control (trimming on): thinking blocks dropped={LEDGER.dropped}")
    if not LEDGER.dropped:
        failures.append("control: re-enabling trimming was not detected -- the "
                        "prefix check has no teeth")

    # A refusing seat and a fallback-served seat in one otherwise normal game.
    LEDGER.__init__()
    log = run_game(RunConfig(
        seed=5, model=spec("claude-opus-5-5"), attack_enabled=True,
        seat_models={3: spec("claude-opus-5-5-refuses"), 4: spec("claude-opus-5-5-fallback")},
    ))
    refused = [s for t, tr in steps(log) if t["player_id"] == 3 for s in tr]
    served = {s.get("served_by") for t, tr in steps(log) if t["player_id"] == 4 for s in tr}
    print(f"refusing seat: steps={[s['action'] for s in refused][:4]}...  "
          f"fallback seat served_by={served}")
    if log["outcome"]["crashed"]:
        failures.append(f"refusal/fallback game crashed: {log['outcome']['crash_reason']}")
    if not refused or any(s["block_reason"] not in ("refusal", "fallback") for s in refused):
        failures.append("a refusal was not recorded as a refusal")
    if any(s["action"] == "<refusal>" for s in refused) and \
            sum(1 for s in refused if s["action"] == "<refusal>") != \
            sum(1 for t, _ in steps(log) if t["player_id"] == 3):
        failures.append("a refused turn asked the model again")
    if served != {"claude-opus-5"}:
        failures.append(f"fallback turns not attributed: {served}")
    for v in LEDGER.violations[:3]:
        failures.append(f"refusal/fallback game: API would reject: {v}")

    # Probe, a transient 529, a model that takes temperature, and error hints.
    LEDGER.__init__()
    LEDGER.fail_next = 2
    result = probe_model(build_client(spec("claude-opus-5-5-flaky")))
    print(f"probe(claude-opus-5-5 after 2x529): reachable={result.reachable} "
          f"tool_mode={result.tool_mode} temperature_stable={result.temperature_stable}")
    if not (result.reachable and result.tool_mode == "native"):
        failures.append(f"probe: {result.error or result.notes}")
    LEDGER.__init__()
    build_client(spec("claude-haiku-4-5")).chat([{"role": "user", "content": "hi"}],
                                                temperature=0.3)
    sent = LEDGER.requests[-1]
    if sent.get("temperature") != 0.3 or "thinking" in sent:
        failures.append(f"haiku request shape wrong: {sorted(sent)}")
    for model, key, expect in (("claude-nope", "sk-ant-x", "Unknown model"),
                               ("claude-opus-5-5", "bad", "rejected")):
        try:
            build_client({**spec(model), "api_key": key}).chat(
                [{"role": "user", "content": "hi"}])
            failures.append(f"{model}/{key}: no error raised")
        except ProviderError as exc:
            print(f"error mapping [{model} + {key}]: {exc.hint}")
            if expect not in (exc.hint or ""):
                failures.append(f"{model}: hint did not mention {expect!r}")

    server.shutdown()
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nthe direct-Claude path is sound: thinking replayed intact, no history "
          "edits, no rejected requests, refusals and fallbacks attributed")
    return 0


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        raise SystemExit(_check() or _check_claude())
    serve()
    print("fake gateway on http://127.0.0.1:8900/v1  (ctrl-c to stop)")
    threading.Event().wait()
