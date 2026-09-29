"""Claude through `claude -p` -- the Claude Code CLI in print mode.

Runs on the Claude subscription the CLI is logged in with, instead of an API
key billed per token. The CLI is used as a single-step completion engine: one
harness step, one `claude -p` call, tools off, one turn. The ReAct loop stays
the harness's own -- the CLI is an agent framework, and letting it run its own
loop would hand it exactly the things being measured.

Every setting below closes a way the CLI would otherwise change what the
agents see, or who pays:

* **Billing.** An `ANTHROPIC_API_KEY` in the environment takes precedence over
  the subscription login, so it is removed from the child's environment.
  (`--bare` is not used for the same reason: it accepts nothing *but* a key.)
* **Nothing from the machine leaks into the prompt.** Without
  `--setting-sources ""` the CLI loads `CLAUDE.md` files into every call -- the
  project's, the user's -- and every agent would silently be playing under
  instructions nobody recorded. Tested: a CLAUDE.md in the working directory
  came through despite `--system-prompt`. The child also runs in an empty
  directory of its own.
* **No inherited session.** Launched from inside a Claude Code session, the
  child inherits that session's variables, including its effort level. Those
  are stripped, and effort is always passed explicitly.
* **One model call per step.** The CLI makes auxiliary calls of its own by
  default; `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` turns them off.
* **Real turn boundaries.** A multi-step turn is one CLI *session*, continued
  with `--resume`, so earlier steps reach the model as real user/assistant
  messages. Flattening the history into one block of text instead would let
  a speech that forges "[assistant]" cross a role boundary -- a new injection
  surface created by the transport, measured as if it were the model's.

What cannot be removed, and is therefore recorded in `describe()`: the CLI
prefixes the system prompt with an Agent-SDK identity line and adds its own
environment, model and date lines. They are the same for every seat and arm.

Tool calling uses the harness's JSON protocol (`tool_mode="json_prompt"`),
the same path as any relay model without native function calling.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from .base import (
    JSON_TOOL_INSTRUCTIONS,
    LLMClient,
    LLMResponse,
    ProviderError,
    ProviderExhausted,
    describe_tools_for_prompt,
    parse_json_action,
)

# Variables that change who pays, or tie the child to a parent session.
_STRIP_EXACT = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDECODE",
    "CLAUDE_EFFORT",
    "CLAUDE_PID",
    "CLAUDE_AFTER_LAST_COMPACT",
    "CLAUDE_AUTO_BACKGROUND_TASKS",
    "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE",
    "CLAUDE_ADDITIONAL_DIRECTORIES",
}
_STRIP_PREFIXES = (
    "CLAUDE_CODE_SESSION_",
    "CLAUDE_CODE_MESSAGING_",
    "CLAUDE_CODE_ADDITIONAL_DIRECTORIES",
    "CLAUDE_CODE_USER_EMAIL",
    "CLAUDE_CODE_TEE_",
    "CLAUDE_CODE_DIAGNOSTICS_",
    "CLAUDE_CODE_BG_",
    "CLAUDE_CODE_CHILD_",
    "CLAUDE_CODE_DEBUG",
)

# Adaptive-thinking models take --effort; older ones reject it.
_EFFORT_PREFIXES = ("claude-fable", "claude-mythos", "claude-opus-5", "claude-sonnet-5",
                    "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
                    "claude-sonnet-4-6", "opus", "sonnet")

_LIMIT_MARKERS = ("usage limit", "hit your limit", "limit reached", "rate limit",
                  "quota", "out of extra usage")


def child_env(base: dict | None = None) -> dict:
    env = dict(os.environ if base is None else base)
    for key in list(env):
        if key in _STRIP_EXACT or key.startswith(_STRIP_PREFIXES):
            del env[key]
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    return env


class ClaudeCLIClient(LLMClient):
    tool_mode = "json_prompt"
    # A turn is a CLI session; the loop may only ever append to it.
    append_only = True
    # The CLI takes no sampling parameters.
    accepts_sampling = False

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        display_name: str | None = None,
        effort: str = "medium",
        executable: str | None = None,
        min_timeout: float = 180.0,
        keep_sessions: int = 64,
    ):
        exe = executable or os.getenv("CLAUDE_CLI") or "claude"
        resolved = shutil.which(exe)
        if resolved is None:
            raise ProviderError(
                f"cannot find the Claude Code CLI ({exe!r})",
                hint="Install Claude Code and log in with `claude` once, or point "
                     "CLAUDE_CLI at the executable.",
            )
        self.executable = resolved
        self.model = model
        self.name = display_name or model
        self.group = None
        self.effort = effort
        self.uses_effort = model.startswith(_EFFORT_PREFIXES)
        self.min_timeout = min_timeout
        self.keep_sessions = keep_sessions
        # An empty directory of its own: no CLAUDE.md can be discovered from it,
        # and its sessions are easy to find and remove.
        self.workdir = tempfile.mkdtemp(prefix="werewolf-claude-cli-")
        self._sessions: dict[str, tuple[str, int]] = {}  # conversation -> (id, consumed)
        self._order: list[str] = []
        self._dirs: set[str] = set()  # where the CLI filed this client's sessions
        self._lock = threading.Lock()

    def describe(self) -> dict:
        return {
            "provider": "claude_cli",
            "model": self.model,
            "billing": "Claude subscription (claude -p)",
            "effort": self.effort if self.uses_effort else None,
            "sampling": "model default",
            "tool_mode": self.tool_mode,
            "context_policy": "append_only (one CLI session per turn)",
            "injected_by_cli": "Agent-SDK identity prefix; environment, model and "
                               "date lines",
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
        system = "\n\n".join(m["content"] for m in messages
                             if m.get("role") == "system" and m.get("content"))
        if tools:
            system = f"{system}\n\n" + JSON_TOOL_INSTRUCTIONS.format(
                tools=describe_tools_for_prompt(tools))
        body = [m for m in messages if m.get("role") != "system"]
        if not body:
            raise ProviderError("nothing to send: the conversation has no user message")

        key = _conversation_key(system, body[0])
        with self._lock:
            known = self._sessions.get(key)
        if known is None:
            session_id, fresh = str(uuid.uuid4()), True
            new = body
        else:
            session_id, consumed = known
            fresh = False
            new = body[consumed:]
        # The CLI session already holds the model's own replies; only what the
        # harness said since then is sent.
        prompt = "\n\n".join(m.get("content") or "" for m in new
                             if m.get("role") != "assistant" and m.get("content"))
        if not prompt:
            raise ProviderError("nothing new to send in this conversation")

        cmd = [self.executable, "-p",
               "--output-format", "json",
               "--model", self.model,
               "--system-prompt", system,
               "--tools", "",
               "--setting-sources", "",
               "--disable-slash-commands",
               "--max-turns", "1"]
        cmd += ["--session-id", session_id] if fresh else ["--resume", session_id]
        if self.uses_effort:
            cmd += ["--effort", self.effort]

        limit = max(timeout, self.min_timeout)
        started = time.time()
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                  timeout=limit, cwd=self.workdir, env=child_env())
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(f"timed out after {limit}s",
                                hint="Lower effort, or raise the timeout.") from exc
        latency_ms = int((time.time() - started) * 1000)

        data = _parse(proc)
        if data.get("is_error"):
            text = str(data.get("result") or data.get("subtype") or "error")
            if any(m in text.lower() for m in _LIMIT_MARKERS):
                # Not retryable within a game, and must not become abstentions:
                # the game is abandoned and counted as crashed instead.
                raise ProviderExhausted(
                    f"subscription limit: {text[:200]}",
                    hint="The Claude subscription's usage limit was reached. Wait for "
                         "it to reset; games stopped by it are marked crashed.",
                )
            raise ProviderError(f"claude -p failed: {text[:300]}",
                                hint=_hint(text))

        with self._lock:
            self._sessions[key] = (session_id, len(body) + 1)  # + the reply
            if fresh:
                self._order.append(session_id)
                self._prune()

        usage = data.get("usage") or {}
        response = LLMResponse(
            text=(data.get("result") or "").strip(),
            prompt_tokens=int(usage.get("input_tokens", 0))
            + int(usage.get("cache_read_input_tokens", 0))
            + int(usage.get("cache_creation_input_tokens", 0)),
            completion_tokens=int(usage.get("output_tokens", 0)),
            latency_ms=latency_ms,
            finish_reason=data.get("stop_reason") or "end_turn",
            model=_served(data) or self.model,
        )
        response.cache_read_tokens = int(usage.get("cache_read_input_tokens", 0))
        if data.get("stop_reason") == "refusal":
            response.refusal = "unspecified"
            return response
        if response.text:
            response.tool_calls.append(parse_json_action(response.text))
        return response

    # -------------------------------------------------------------- sessions

    def _prune(self) -> None:
        """Turns are short; once a conversation is this far back it is over."""
        while len(self._order) > self.keep_sessions:
            self._dirs |= _delete_session(self._order.pop(0))

    def close(self) -> None:
        with self._lock:
            for sid in self._order:
                self._dirs |= _delete_session(sid)
            self._order.clear()
            self._sessions.clear()
            # One project directory per client; left behind empty, a batch
            # would litter the user's session list with them.
            for d in self._dirs:
                try:
                    os.rmdir(d)
                except OSError:
                    pass  # not empty: something else lives there, leave it
            self._dirs.clear()
        shutil.rmtree(self.workdir, ignore_errors=True)

    def __del__(self):  # best effort; close() is the reliable path
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def _conversation_key(system: str, first: dict) -> str:
    import hashlib

    return hashlib.sha256(f"{system}\x00{first.get('content') or ''}".encode()).hexdigest()


def _delete_session(session_id: str) -> set[str]:
    """Remove a finished conversation; return the directories it was in."""
    root = Path(os.getenv("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "projects"
    dirs = set()
    for path in glob.glob(str(root / "*" / f"{session_id}.jsonl")):
        dirs.add(os.path.dirname(path))
        try:
            os.remove(path)
        except OSError:
            pass
    return dirs


def _parse(proc: subprocess.CompletedProcess) -> dict:
    out = (proc.stdout or "").strip()
    try:
        data = json.loads(out.splitlines()[-1] if out else "")
    except (json.JSONDecodeError, IndexError):
        err = (proc.stderr or out or f"exit {proc.returncode}").strip()
        low = err.lower()
        if any(m in low for m in _LIMIT_MARKERS):
            raise ProviderExhausted(f"subscription limit: {err[:200]}",
                                    hint="The Claude subscription's usage limit was "
                                         "reached.") from None
        raise ProviderError(f"claude -p returned no result: {err[:300]}",
                            hint=_hint(err)) from None
    if not isinstance(data, dict):
        raise ProviderError(f"unexpected claude -p output: {out[:200]}")
    return data


def _served(data: dict) -> str | None:
    """The model that answered, from the CLI's own accounting."""
    usage = data.get("modelUsage") or {}
    names = [v.get("canonicalModel") or k for k, v in usage.items()]
    return names[-1] if names else None


def _hint(text: str) -> str:
    low = (text or "").lower()
    if "auth" in low or "log in" in low or "login" in low:
        return "The CLI is not logged in. Run `claude` once and sign in with your subscription."
    if "model" in low:
        return "Unknown model. Use an ID such as claude-opus-5-5 or claude-sonnet-5-5."
    return "See the raw message; running the same command by hand usually shows why."
