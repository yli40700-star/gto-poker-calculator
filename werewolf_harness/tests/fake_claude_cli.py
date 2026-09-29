"""A stand-in for the `claude` executable, for testing `ClaudeCLIClient`.

Behaves like `claude -p --output-format json` closely enough to drive whole
games: prompt on stdin, `--session-id` to start a conversation and `--resume`
to continue it, one JSON result on stdout. Decisions come from the offline
player, answering in the harness's JSON protocol.

It also checks every invocation against the settings the client exists to
enforce, and appends each breach to `$FAKE_CLAUDE_STATE/violations.jsonl`:
a key in the environment (the subscription would not be what pays), a missing
`--setting-sources ""` (CLAUDE.md files would reach the prompt), an inherited
effort level, tools left on, more than one turn.

    FAKE_CLAUDE_STATE=/tmp/x python -m werewolf_harness.tests.fake_claude_cli -p ...
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ..harness.providers.mock import MockClient

LIMITED = "claude-opus-5-5-limited"  # answers N times, then reports the usage cap


def main(argv: list[str]) -> int:
    state = Path(os.environ.get("FAKE_CLAUDE_STATE", "/tmp/fake-claude"))
    (state / "sessions").mkdir(parents=True, exist_ok=True)
    args = _args(argv)
    problems = _check(args)
    for p in problems:
        with open(state / "violations.jsonl", "a") as fh:
            fh.write(json.dumps(p) + "\n")

    with open(state / "calls.jsonl", "a") as fh:
        fh.write(json.dumps({"model": args.get("--model"),
                             "resume": "--resume" in args}) + "\n")

    model = args.get("--model", "")
    if model == LIMITED:
        count = state / "limited.count"
        n = int(count.read_text()) if count.exists() else 0
        count.write_text(str(n + 1))
        if n >= int(os.environ.get("FAKE_CLAUDE_LIMIT_AFTER", "5")):
            print(json.dumps({"type": "result", "subtype": "success", "is_error": True,
                              "result": "Claude AI usage limit reached|1760000000"}))
            return 1

    prompt = sys.stdin.read()
    sid = args.get("--session-id") or args.get("--resume")
    path = state / "sessions" / f"{sid}.json"
    if "--resume" in args:
        if not path.exists():
            print(json.dumps({"type": "result", "is_error": True,
                              "result": f"No conversation found with session ID: {sid}"}))
            return 1
        history = json.loads(path.read_text())
    else:
        if path.exists():
            print(json.dumps({"type": "result", "is_error": True,
                              "result": f"Session ID {sid} is already in use"}))
            return 1
        history = []
    history.append({"role": "user", "content": prompt})

    decision = MockClient(seed=0).chat(
        [{"role": "system", "content": args.get("--system-prompt", "")}] + history)
    call = decision.tool_calls[0] if decision.tool_calls else None
    reply = json.dumps({"thought": "Weighing who contradicted themselves.",
                        "action": call.name if call else "speak",
                        "args": call.arguments if call else {}})
    history.append({"role": "assistant", "content": reply})
    path.write_text(json.dumps(history))

    served = model.replace("-limited", "")
    print(json.dumps({
        "type": "result", "subtype": "success", "is_error": False,
        "result": reply, "session_id": sid, "stop_reason": "end_turn",
        "usage": {"input_tokens": decision.prompt_tokens, "output_tokens":
                  decision.completion_tokens, "cache_read_input_tokens": 0,
                  "cache_creation_input_tokens": 0},
        "modelUsage": {served: {"canonicalModel": served,
                                "inputTokens": decision.prompt_tokens}},
        "total_cost_usd": 0.0,
    }))
    return 0


def _args(argv: list[str]) -> dict:
    out: dict = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-p", "--print", "--disable-slash-commands"):
            out[a] = True
            i += 1
        elif a.startswith("--") and i + 1 < len(argv):
            out[a] = argv[i + 1]
            i += 2
        else:
            i += 1
    return out


def _check(args: dict) -> list[str]:
    env = os.environ
    problems = []
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        if env.get(var):
            problems.append(f"{var} reached the child: the API key would be billed, "
                            "not the subscription")
    if env.get("CLAUDE_EFFORT"):
        problems.append("CLAUDE_EFFORT inherited from a parent session")
    if env.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC") != "1":
        problems.append("auxiliary model calls left on")
    if args.get("--setting-sources") != "":
        problems.append("--setting-sources '' missing: CLAUDE.md files reach the prompt")
    if args.get("--tools") != "":
        problems.append("built-in tools left on")
    if args.get("--max-turns") != "1":
        problems.append("the CLI may run its own multi-turn loop")
    if args.get("--output-format") != "json":
        problems.append("output is not JSON")
    if "--bare" in args:
        problems.append("--bare ignores the subscription login")
    model = args.get("--model", "")
    if model.startswith(("claude-opus-5", "claude-sonnet-5")) and "--effort" not in args:
        problems.append("effort not set explicitly")
    if Path.cwd().joinpath("CLAUDE.md").exists():
        problems.append("working directory contains a CLAUDE.md")
    return problems


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
