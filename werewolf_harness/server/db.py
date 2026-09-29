"""SQLite storage.

Four tables, no ORM, and the game log is stored as a single JSON blob rather
than shredded into rows: nothing here needs a relational query, and metric
extraction is an offline pass over the logs. Normalising it would buy nothing
and would make the log schema harder to extend.

The one rule with teeth: `providers.api_key` is written by the server and never
read back out over the API. Everything the browser sees is masked.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

DEFAULT_PATH = Path("werewolf_harness.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS providers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    base_url TEXT NOT NULL,
    api_key TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'openai_compat',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    provider_id TEXT NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
    display_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    group_name TEXT,
    temperature REAL NOT NULL DEFAULT 0.7,
    max_tokens INTEGER NOT NULL DEFAULT 700,
    supports_tools INTEGER,
    tool_mode TEXT NOT NULL DEFAULT 'native',
    notes TEXT DEFAULT '',
    probe_json TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS games (
    id TEXT PRIMARY KEY,
    experiment_id TEXT,
    seed INTEGER NOT NULL,
    label TEXT,
    config_json TEXT NOT NULL,
    log_json TEXT NOT NULL,
    outcome_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL,
    progress_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_games_experiment ON games(experiment_id);
"""


def connect(path: str | Path = DEFAULT_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn) -> None:
    """Bring a database created by an earlier version up to date, in place.
    Additive only: a column is added with a default, nothing is rewritten."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(providers)")}
    if "kind" not in cols:
        conn.execute("ALTER TABLE providers ADD COLUMN kind TEXT NOT NULL "
                     "DEFAULT 'openai_compat'")
        conn.execute("UPDATE providers SET kind = 'anthropic' "
                     "WHERE base_url LIKE '%api.anthropic.com%'")
        conn.commit()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def mask_key(key: str) -> str:
    """What the browser is allowed to see. Never the key itself."""
    if not key:
        return ""
    return f"{key[:3]}****{key[-4:]}" if len(key) > 10 else "****"


# ------------------------------------------------------------- providers

def add_provider(conn, name: str, base_url: str, api_key: str,
                 kind: str | None = None) -> dict:
    if kind is None:
        kind = "anthropic" if "api.anthropic.com" in base_url else "openai_compat"
    if kind not in ("anthropic", "openai_compat", "claude_cli"):
        raise ValueError(f"unknown provider kind {kind!r}")
    pid = new_id("prov")
    conn.execute(
        "INSERT INTO providers (id, name, base_url, api_key, kind, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (pid, name, base_url.rstrip("/"), api_key, kind, time.time()),
    )
    conn.commit()
    return get_provider(conn, pid)


def get_provider(conn, provider_id: str, with_key: bool = False) -> dict | None:
    row = conn.execute("SELECT * FROM providers WHERE id = ?", (provider_id,)).fetchone()
    if row is None:
        return None
    return _provider_dict(row, with_key)


def list_providers(conn) -> list[dict]:
    rows = conn.execute("SELECT * FROM providers ORDER BY created_at").fetchall()
    return [_provider_dict(r, False) for r in rows]


def delete_provider(conn, provider_id: str) -> None:
    conn.execute("DELETE FROM providers WHERE id = ?", (provider_id,))
    conn.commit()


def _provider_dict(row, with_key: bool) -> dict:
    out = {
        "id": row["id"],
        "name": row["name"],
        "base_url": row["base_url"],
        "kind": row["kind"],
        "api_key_masked": mask_key(row["api_key"]),
        "created_at": row["created_at"],
    }
    if with_key:
        out["api_key"] = row["api_key"]
    return out


# ---------------------------------------------------------------- models

def add_model(conn, provider_id: str, display_name: str, model_name: str,
              group: str | None = None, temperature: float = 0.7,
              max_tokens: int = 700, tool_mode: str = "native",
              notes: str = "") -> dict:
    mid = new_id("model")
    conn.execute(
        "INSERT INTO models (id, provider_id, display_name, model_name, group_name, "
        "temperature, max_tokens, supports_tools, tool_mode, notes, probe_json, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL, ?)",
        (mid, provider_id, display_name, model_name, group, temperature, max_tokens,
         tool_mode, notes, time.time()),
    )
    conn.commit()
    return get_model(conn, mid)


def get_model(conn, model_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    return _model_dict(row) if row else None


def list_models(conn) -> list[dict]:
    rows = conn.execute("SELECT * FROM models ORDER BY created_at").fetchall()
    return [_model_dict(r) for r in rows]


def delete_model(conn, model_id: str) -> None:
    conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
    conn.commit()


def save_probe(conn, model_id: str, probe: dict) -> None:
    """The probe result decides `tool_mode`, not the person filling the form."""
    conn.execute(
        "UPDATE models SET supports_tools = ?, tool_mode = ?, probe_json = ? "
        "WHERE id = ?",
        (int(bool(probe.get("native_tools"))), probe.get("tool_mode", "json_prompt"),
         json.dumps(probe), model_id),
    )
    conn.commit()


def _model_dict(row) -> dict:
    return {
        "id": row["id"],
        "provider_id": row["provider_id"],
        "display_name": row["display_name"],
        "model_name": row["model_name"],
        "group": row["group_name"],
        "temperature": row["temperature"],
        "max_tokens": row["max_tokens"],
        "supports_tools": None if row["supports_tools"] is None else bool(row["supports_tools"]),
        "tool_mode": row["tool_mode"],
        "notes": row["notes"],
        "probe": json.loads(row["probe_json"]) if row["probe_json"] else None,
    }


# ----------------------------------------------------------------- games

def save_game(conn, log: dict, label: str = "") -> str:
    conn.execute(
        "INSERT OR REPLACE INTO games (id, experiment_id, seed, label, config_json, "
        "log_json, outcome_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            log["game_id"],
            log.get("experiment_id"),
            log["seed"],
            label,
            json.dumps(log["config"], ensure_ascii=False),
            json.dumps(log, ensure_ascii=False),
            json.dumps(log["outcome"], ensure_ascii=False),
            time.time(),
        ),
    )
    conn.commit()
    return log["game_id"]


def get_game(conn, game_id: str) -> dict | None:
    row = conn.execute("SELECT log_json FROM games WHERE id = ?", (game_id,)).fetchone()
    return json.loads(row["log_json"]) if row else None


def list_games(conn, experiment_id: str | None = None, limit: int = 100) -> list[dict]:
    if experiment_id:
        rows = conn.execute(
            "SELECT id, experiment_id, seed, label, config_json, outcome_json, created_at "
            "FROM games WHERE experiment_id = ? ORDER BY created_at DESC LIMIT ?",
            (experiment_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, experiment_id, seed, label, config_json, outcome_json, created_at "
            "FROM games ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "game_id": r["id"],
            "experiment_id": r["experiment_id"],
            "seed": r["seed"],
            "label": r["label"],
            "config": json.loads(r["config_json"]),
            "outcome": json.loads(r["outcome_json"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]


# ------------------------------------------------------------ experiments

def create_experiment(conn, name: str, config: dict) -> dict:
    eid = new_id("exp")
    conn.execute(
        "INSERT INTO experiments (id, name, config_json, status, progress_json, "
        "created_at) VALUES (?, ?, ?, 'queued', ?, ?)",
        (eid, name, json.dumps(config, ensure_ascii=False),
         json.dumps({"done": 0, "total": 0, "crashed": 0, "tokens": 0}), time.time()),
    )
    conn.commit()
    return get_experiment(conn, eid)


def update_experiment(conn, experiment_id: str, status: str | None = None,
                      progress: dict | None = None) -> None:
    if status is not None:
        conn.execute("UPDATE experiments SET status = ? WHERE id = ?",
                     (status, experiment_id))
    if progress is not None:
        conn.execute("UPDATE experiments SET progress_json = ? WHERE id = ?",
                     (json.dumps(progress), experiment_id))
    conn.commit()


def get_experiment(conn, experiment_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM experiments WHERE id = ?",
                       (experiment_id,)).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "config": json.loads(row["config_json"]),
        "status": row["status"],
        "progress": json.loads(row["progress_json"]),
        "created_at": row["created_at"],
    }


def list_experiments(conn) -> list[dict]:
    rows = conn.execute("SELECT id FROM experiments ORDER BY created_at DESC").fetchall()
    return [get_experiment(conn, r["id"]) for r in rows]
