"""The direct-Claude path.

The end-to-end test drives full games through the real `anthropic` SDK against
a local server that enforces the Messages API's rules. The unit tests pin the
translation rules that keep a request valid.
"""

from __future__ import annotations

import sqlite3

import pytest

from werewolf_harness.harness.providers import build_client, provider_kind
from werewolf_harness.harness.providers.anthropic_client import (
    NOT_EXECUTED,
    to_anthropic,
    to_anthropic_tool,
)
from werewolf_harness.harness.providers.openai_compat import _wire


def test_full_games_through_the_sdk_break_no_api_rule():
    """Thinking replayed intact, no history edits, no request the API would
    reject, refusals and fallbacks attributed -- and a negative control showing
    the history check actually catches the old trimming."""
    pytest.importorskip("anthropic")
    from werewolf_harness.tests.fake_gateway import _check_claude

    assert _check_claude(port=8921) == 0


# ------------------------------------------------------------ translation

def test_system_is_lifted_out_and_tool_results_become_user_blocks():
    system, wire = to_anthropic([
        {"role": "system", "content": "You are player 3, role: seer"},
        {"role": "user", "content": "situation"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t1", "type": "function",
             "function": {"name": "query_history", "arguments": '{"player_id": 5}'}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "record"},
    ])
    assert system == "You are player 3, role: seer"
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    assert wire[1]["content"][0] == {"type": "tool_use", "id": "t1",
                                     "name": "query_history", "input": {"player_id": 5}}
    assert wire[2]["content"] == [{"type": "tool_result", "tool_use_id": "t1",
                                   "content": "record"}]


def test_native_content_is_replayed_verbatim_not_rebuilt():
    """Thinking blocks must go back exactly as they came."""
    blocks = [{"type": "thinking", "thinking": "hm", "signature": "sig"},
              {"type": "tool_use", "id": "t1", "name": "vote", "input": {"target_id": 4}}]
    _, wire = to_anthropic([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "native_content": blocks,
         "tool_calls": [{"id": "t1", "function": {"name": "vote", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "ok"},
    ])
    assert wire[1]["content"] == blocks


def test_an_unanswered_parallel_call_gets_an_explicit_result():
    """The loop acts on one call per step. A second call left without a result
    is a 400; it gets a 'not executed' result instead of being dropped."""
    blocks = [{"type": "tool_use", "id": "a", "name": "query_votes", "input": {}},
              {"type": "tool_use", "id": "b", "name": "query_votes", "input": {}}]
    _, wire = to_anthropic([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "native_content": blocks},
        {"role": "tool", "tool_call_id": "a", "content": "result"},
        {"role": "user", "content": "[guard] next"},
    ])
    results = [b for b in wire[2]["content"] if b["type"] == "tool_result"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert results[1]["content"] == NOT_EXECUTED and results[1]["is_error"]
    # results first, then text
    assert [b["type"] for b in wire[2]["content"]] == ["tool_result", "tool_result", "text"]


def test_an_empty_assistant_turn_is_dropped_not_sent():
    _, wire = to_anthropic([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "[guard] no usable tool call"},
    ])
    assert [m["role"] for m in wire] == ["user"]
    assert len(wire[0]["content"]) == 2


def test_translation_is_deterministic():
    """Same history, same bytes: what keeps an append-only history append-only
    on the wire (and the prompt cache warm)."""
    history = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert to_anthropic(history) == to_anthropic(history)


def test_tool_schema_translation():
    tool = to_anthropic_tool({"type": "function", "function": {
        "name": "vote", "description": "d",
        "parameters": {"type": "object", "properties": {"target_id": {"type": "integer"}}}}})
    assert tool == {"name": "vote", "description": "d", "input_schema": {
        "type": "object", "properties": {"target_id": {"type": "integer"}}}}


def test_relay_requests_never_carry_claude_fields():
    """A strict gateway rejects a message field it does not know."""
    sent = _wire([{"role": "assistant", "content": None, "tool_calls": [],
                   "native_content": [{"type": "thinking"}]}])
    assert sent == [{"role": "assistant", "content": None, "tool_calls": []}]


# --------------------------------------------------------------- routing

def test_the_provider_decides_the_client_never_the_model_name():
    """A relay serves claude-* names too; those must stay on the relay."""
    assert provider_kind({"base_url": "https://api.anthropic.com"}) == "anthropic"
    assert provider_kind({"base_url": "https://api.aipaibox.com/v1",
                          "model_name": "claude-opus-5-5"}) == "openai_compat"
    assert provider_kind({"provider": "anthropic", "base_url": "http://127.0.0.1"}) == "anthropic"


def test_request_shape_follows_the_model():
    pytest.importorskip("anthropic")
    opus = build_client({"provider": "anthropic", "model_name": "claude-opus-5-5",
                         "api_key": "sk-ant-x"})
    haiku = build_client({"provider": "anthropic", "model_name": "claude-haiku-4-5",
                          "api_key": "sk-ant-x"})
    assert opus.describe()["sampling"] == "model default"
    assert opus.describe()["fallbacks"] == "default"
    assert opus.append_only
    assert haiku.describe()["sampling"] == "temperature"
    assert haiku.describe()["thinking"] == "off"
    assert haiku.describe()["fallbacks"] == "off"  # no server-side fallback for it
    off = build_client({"provider": "anthropic", "model_name": "claude-opus-5-5",
                        "api_key": "sk-ant-x", "fallbacks": "off"})
    assert off.describe()["fallbacks"] == "off"


def test_a_base_url_with_v1_is_not_doubled():
    pytest.importorskip("anthropic")
    client = build_client({"provider": "anthropic", "model_name": "claude-opus-5-5",
                           "api_key": "sk-ant-x", "base_url": "https://api.anthropic.com/v1"})
    assert client.base_url == "https://api.anthropic.com"


# ---------------------------------------------------------------- storage

def test_an_old_database_gains_the_provider_kind_column(tmp_path):
    from werewolf_harness.server import db as dbmod

    path = tmp_path / "old.db"
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE providers (id TEXT PRIMARY KEY, name TEXT NOT NULL, "
                "base_url TEXT NOT NULL, api_key TEXT NOT NULL, created_at REAL NOT NULL)")
    raw.execute("INSERT INTO providers VALUES ('p1','relay','https://r.example/v1','sk-a',0)")
    raw.execute("INSERT INTO providers VALUES ('p2','claude','https://api.anthropic.com','sk-ant-b',0)")
    raw.commit()
    raw.close()

    conn = dbmod.connect(path)
    kinds = {p["name"]: p["kind"] for p in dbmod.list_providers(conn)}
    assert kinds == {"relay": "openai_compat", "claude": "anthropic"}
