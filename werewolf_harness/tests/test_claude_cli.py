"""`claude -p` as a model backend, driven through a stand-in executable that
checks every invocation (see `fake_claude_cli`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from werewolf_harness.evalkit.runner import RunConfig, run_game
from werewolf_harness.harness.providers import claude_cli
from werewolf_harness.harness.providers.claude_cli import child_env

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def fake_cli(tmp_path, monkeypatch):
    """A `claude` executable backed by the offline player, plus a hostile
    environment: a key that would move billing to the API, and an effort
    level inherited from a parent session."""
    state = tmp_path / "state"
    exe = tmp_path / "claude"
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" -m '
                   f'werewolf_harness.tests.fake_claude_cli "$@"\n')
    exe.chmod(0o755)
    monkeypatch.setenv("FAKE_CLAUDE_STATE", str(state))
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-would-be-billed")
    monkeypatch.setenv("CLAUDE_EFFORT", "max")
    return exe, state


def _spec(exe, model="claude-opus-5-5"):
    return {"provider": "claude_cli", "model_name": model, "display_name": model,
            "executable": str(exe), "tool_mode": "json_prompt"}


def _lines(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_a_full_game_through_the_cli_breaks_none_of_its_settings(fake_cli):
    exe, state = fake_cli
    log = run_game(RunConfig(seed=5, model=_spec(exe), guard_layers=("L1", "L2"),
                             attack_enabled=True))

    assert not log["outcome"]["crashed"], log["outcome"].get("crash_reason")
    assert _lines(state / "violations.jsonl") == []

    calls = _lines(state / "calls.jsonl")
    assert any(c["resume"] for c in calls), "multi-step turns must continue a session"
    assert any(not c["resume"] for c in calls)

    settings = log["config"]["provider_settings"]["claude-opus-5-5"]
    assert settings["billing"].startswith("Claude subscription")
    assert settings["effort"] == "medium"
    assert log["outcome"]["total_cost_usd"] == 0.0
    thoughts = [s["thought"] for r in log["rounds"] for a in r["agents"]
                for s in a["react_trace"]]
    assert any(t.startswith("Weighing") for t in thoughts)


def test_the_checks_have_teeth(fake_cli, monkeypatch):
    """Negative control: pass the environment through untouched and the
    stand-in must report the billing leak and the inherited effort."""
    exe, state = fake_cli
    monkeypatch.setattr(claude_cli, "child_env", lambda base=None: dict(__import__("os").environ))
    client = claude_cli.ClaudeCLIClient(model="claude-opus-5-5", executable=str(exe))
    try:
        client.chat([{"role": "system", "content": "You are player 1, role: villager"},
                     {"role": "user", "content": "=== ROUND 1\nAlive: [1, 2, 3]"}])
    finally:
        client.close()
    problems = " ".join(_lines(state / "violations.jsonl"))
    assert "ANTHROPIC_API_KEY" in problems
    assert "CLAUDE_EFFORT" in problems


def test_a_usage_limit_abandons_the_game_instead_of_recording_abstentions(fake_cli, monkeypatch):
    exe, _ = fake_cli
    monkeypatch.setenv("FAKE_CLAUDE_LIMIT_AFTER", "6")
    log = run_game(RunConfig(seed=5, model=_spec(exe, "claude-opus-5-5-limited"),
                             attack_enabled=True))
    assert log["outcome"]["crashed"]
    assert "subscription limit" in log["outcome"]["crash_reason"]


def test_the_child_environment():
    env = child_env({"PATH": "/bin", "HOME": "/h", "ANTHROPIC_API_KEY": "k",
                     "CLAUDE_EFFORT": "max", "CLAUDE_CODE_SESSION_ID": "s",
                     "CLAUDE_CODE_USER_EMAIL": "a@b", "CLAUDECODE": "1"})
    assert env == {"PATH": "/bin", "HOME": "/h",
                   "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}


def test_close_removes_sessions_and_their_empty_directory(fake_cli, tmp_path, monkeypatch):
    """A batch creates a client per game; each must leave nothing in the
    user's session list behind it."""
    exe, _ = fake_cli
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
    project = tmp_path / "cfg" / "projects" / "-tmp-werewolf-claude-cli-x"
    project.mkdir(parents=True)
    (project / "abc.jsonl").write_text("{}")
    client = claude_cli.ClaudeCLIClient(model="claude-opus-5-5", executable=str(exe))
    client._order.append("abc")
    client.close()
    assert not project.exists()
