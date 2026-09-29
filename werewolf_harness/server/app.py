"""Dashboard backend.

Small on purpose. Its whole job is: hold the API key so the browser never has
to, run games and batches off the request thread, and hand the replay page a
game log.

The key never crosses back over the wire. The browser can create a provider,
see `sk-a****3f2a`, test it and delete it -- it can never read it back, and it
never talks to the gateway itself. That is not ceremony: a repo meant to be
read by strangers must not be one careless commit away from leaking a token.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..attacks import CATEGORIES
from ..evalkit import metrics
from ..evalkit.runner import RunConfig, run_game
from ..harness.providers import ProviderError, build_client, probe_model
from . import db

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
DB_PATH = os.getenv("WEREWOLF_DB", str(db.DEFAULT_PATH))

app = FastAPI(title="Werewolf agent-harness dashboard")
conn = db.connect(DB_PATH)

# Live game event queues, for the SSE replay/spectate stream.
_streams: dict[str, queue.Queue] = {}
_experiments: dict[str, dict] = {}
_lock = threading.Lock()


# ------------------------------------------------------------- schemas

class ProviderIn(BaseModel):
    name: str
    base_url: str
    api_key: str = ""
    # "claude_cli", "anthropic" or "openai_compat"; inferred from base_url
    # when omitted. Only claude_cli needs no key: the CLI's own login pays.
    kind: str | None = None


class ModelIn(BaseModel):
    provider_id: str
    display_name: str
    model_name: str
    group: str | None = None
    temperature: float = 0.7
    max_tokens: int = 700
    tool_mode: str = "native"
    notes: str = ""


class GameIn(BaseModel):
    seed: int = 1
    model_id: str | None = None
    guard_layers: list[str] = []
    evidence_forced: bool = False
    attack_enabled: bool = True
    benign_persuasion: bool = False
    attack_payload: str | None = None
    max_react_steps: int = 8
    human_seat: int | None = None


class ExperimentIn(BaseModel):
    name: str = "ablation"
    seeds: int = 10
    model_id: str | None = None
    arms: list[dict] = []
    workers: int = 3
    include_benign: bool = True


# ------------------------------------------------------------ providers

@app.get("/api/providers")
def get_providers():
    return db.list_providers(conn)


@app.post("/api/providers")
def post_provider(body: ProviderIn):
    if body.kind != "claude_cli" and not body.api_key:
        raise HTTPException(422, "an API key is required for this provider")
    if body.kind == "claude_cli":
        body.base_url = body.base_url or "claude -p"
    try:
        provider = db.add_provider(conn, body.name, body.base_url, body.api_key,
                                   body.kind)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if provider["kind"] == "openai_compat" and not body.base_url.rstrip("/").endswith("/v1"):
        # Not fatal (gateways differ) but it is the single most common setup
        # mistake, so it comes back as a warning rather than as a 404 later.
        provider["warning"] = (
            "base_url does not end in /v1; most OpenAI-compatible gateways "
            "require it"
        )
    return provider


@app.delete("/api/providers/{provider_id}")
def remove_provider(provider_id: str):
    db.delete_provider(conn, provider_id)
    return {"deleted": provider_id}


# --------------------------------------------------------------- models

@app.get("/api/models")
def get_models():
    return db.list_models(conn)


@app.post("/api/models")
def post_model(body: ModelIn):
    if db.get_provider(conn, body.provider_id) is None:
        raise HTTPException(404, "no such provider")
    return db.add_model(
        conn, body.provider_id, body.display_name, body.model_name, body.group,
        body.temperature, body.max_tokens, body.tool_mode, body.notes,
    )


@app.delete("/api/models/{model_id}")
def remove_model(model_id: str):
    db.delete_model(conn, model_id)
    return {"deleted": model_id}


@app.post("/api/models/{model_id}/probe")
def probe(model_id: str, check_temperature: bool = False):
    """Reachability, native tool calling, and the fallback decision.

    Errors come back with an actionable hint rather than the gateway's raw
    text, because the three common failures here are all fixable settings.
    """
    model = db.get_model(conn, model_id)
    if model is None:
        raise HTTPException(404, "no such model")
    provider = db.get_provider(conn, model["provider_id"], with_key=True)
    if provider is None:
        raise HTTPException(404, "the model's provider no longer exists")
    try:
        client = build_client(_model_config(model_id))
    except ProviderError as exc:
        raise HTTPException(400, {"error": str(exc), "hint": exc.hint}) from exc
    result = probe_model(client, check_temperature=check_temperature).to_dict()
    db.save_probe(conn, model_id, result)
    return result


# ---------------------------------------------------------------- games

@app.get("/api/games")
def get_games(experiment_id: str | None = None, limit: int = 100):
    return db.list_games(conn, experiment_id, limit)


@app.get("/api/games/{game_id}")
def get_one_game(game_id: str):
    log = db.get_game(conn, game_id)
    if log is None:
        raise HTTPException(404, "no such game")
    return log


@app.get("/api/games/{game_id}/metrics")
def game_metrics(game_id: str):
    log = db.get_game(conn, game_id)
    if log is None:
        raise HTTPException(404, "no such game")
    return {
        "injection_trials": metrics.injection_trials(log),
        "consistency": metrics.consistency(log),
        "stability": metrics.stability(log),
        "conformity": metrics.conformity(log),
        "hallucination": metrics.hallucination(log),
    }


@app.post("/api/games")
def post_game(body: GameIn):
    cfg = _run_config(body)
    game_id = f"live_{int(time.time() * 1000) % 10_000_000}"
    events: queue.Queue = queue.Queue()
    with _lock:
        _streams[game_id] = events

    def worker():
        def emit(kind, payload):
            events.put({"kind": kind, "payload": payload})

        try:
            log = run_game(cfg, on_event=emit)
            db.save_game(conn, log, label=cfg.label())
            events.put({"kind": "done", "payload": {"game_id": log["game_id"]}})
        except Exception as exc:  # noqa: BLE001
            events.put({"kind": "error", "payload": {"error": str(exc)}})
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()
    return {"stream_id": game_id, "status": "running", "config": cfg.label()}


@app.get("/api/games/{stream_id}/stream")
def stream_game(stream_id: str, request: Request):
    """Server-sent events. One-way push is all this needs, so no websocket."""
    events = _streams.get(stream_id)
    if events is None:
        raise HTTPException(404, "no such live game")

    def generate():
        while True:
            try:
                item = events.get(timeout=30)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if item is None:
                yield "event: end\ndata: {}\n\n"
                break
            yield f"event: {item['kind']}\ndata: {json.dumps(item['payload'], default=str)}\n\n"
        with _lock:
            _streams.pop(stream_id, None)

    return StreamingResponse(generate(), media_type="text/event-stream")


# ----------------------------------------------------------- experiments

@app.get("/api/experiments")
def get_experiments():
    return db.list_experiments(conn)


@app.post("/api/experiments")
def post_experiment(body: ExperimentIn):
    arms = body.arms or [
        {"guard_layers": [], "evidence_forced": False},
        {"guard_layers": ["L1"], "evidence_forced": False},
        {"guard_layers": ["L1", "L2"], "evidence_forced": False},
        {"guard_layers": ["L1", "L2", "L3"], "evidence_forced": False},
        {"guard_layers": ["L1", "L2", "L3"], "evidence_forced": True},
    ]
    experiment = db.create_experiment(conn, body.name, body.model_dump())
    eid = experiment["id"]
    model = _model_config(body.model_id)

    configs = []
    for arm in arms:
        for seed in range(body.seeds):
            base = dict(
                seed=seed,
                model=model,
                guard_layers=tuple(arm.get("guard_layers", [])),
                evidence_forced=bool(arm.get("evidence_forced")),
                experiment_id=eid,
            )
            configs.append(RunConfig(attack_enabled=True, **base))
            if body.include_benign:
                configs.append(RunConfig(benign_persuasion=True, **base))

    state = {"stop": False, "done": 0, "total": len(configs), "crashed": 0, "tokens": 0}
    _experiments[eid] = state

    def worker():
        from ..evalkit.runner import run_batch

        db.update_experiment(conn, eid, status="running",
                             progress={k: state[k] for k in
                                       ("done", "total", "crashed", "tokens")})

        def progress(done, total, log):
            state["done"] = done
            state["crashed"] += bool(log.get("outcome", {}).get("crashed"))
            state["tokens"] += log.get("outcome", {}).get("total_prompt_tokens", 0)
            state["tokens"] += log.get("outcome", {}).get("total_completion_tokens", 0)
            db.save_game(conn, log)
            db.update_experiment(conn, eid, progress={
                "done": done, "total": state["total"],
                "crashed": state["crashed"], "tokens": state["tokens"],
            })

        run_batch(configs, workers=body.workers, on_progress=progress,
                  should_stop=lambda: state["stop"])
        db.update_experiment(conn, eid,
                             status="stopped" if state["stop"] else "finished")

    threading.Thread(target=worker, daemon=True).start()
    return experiment


@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: str):
    experiment = db.get_experiment(conn, experiment_id)
    if experiment is None:
        raise HTTPException(404, "no such experiment")
    return experiment


@app.post("/api/experiments/{experiment_id}/stop")
def stop_experiment(experiment_id: str):
    """Stopping matters: an experiment started with the wrong config is money
    on fire, and it has to be interruptible from the page that started it."""
    state = _experiments.get(experiment_id)
    if state is None:
        raise HTTPException(404, "that experiment is not running here")
    state["stop"] = True
    db.update_experiment(conn, experiment_id, status="stopping")
    return {"stopping": experiment_id}


@app.get("/api/experiments/{experiment_id}/metrics")
def experiment_metrics(experiment_id: str):
    logs = [
        db.get_game(conn, row["game_id"])
        for row in db.list_games(conn, experiment_id, limit=100_000)
    ]
    logs = [log for log in logs if log]
    arms: dict[str, list[dict]] = {}
    for log in logs:
        label = "+".join(log["config"]["guard_layers"]) or "none"
        if log["config"].get("evidence_forced"):
            label += "+E"
        arms.setdefault(label, []).append(log)
    return {
        "experiment_id": experiment_id,
        "games": len(logs),
        "arms": {
            label: metrics.summarise(arm_logs) for label, arm_logs in sorted(arms.items())
        },
    }


# --------------------------------------------------------------- static

@app.get("/api/taxonomy")
def taxonomy():
    return CATEGORIES


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# --------------------------------------------------------------- helpers

def _model_config(model_id: str | None) -> dict:
    """Turn a stored model row into a runner model config, key included.

    This is the only place the key is read, and it never leaves the process.
    """
    if not model_id:
        return {"model_name": "mock"}
    model = db.get_model(conn, model_id)
    if model is None:
        raise HTTPException(404, "no such model")
    provider = db.get_provider(conn, model["provider_id"], with_key=True)
    if provider is None:
        raise HTTPException(404, "the model's provider no longer exists")
    return {
        "model_name": model["model_name"],
        "display_name": model["display_name"],
        "api_key": provider["api_key"],
        "base_url": provider["base_url"],
        "tool_mode": "json_prompt" if provider["kind"] == "claude_cli" else model["tool_mode"],
        "group": model["group"],
        "provider": provider["kind"],
    }


def _run_config(body: GameIn) -> RunConfig:
    return RunConfig(
        seed=body.seed,
        model=_model_config(body.model_id),
        guard_layers=tuple(body.guard_layers),
        evidence_forced=body.evidence_forced,
        attack_enabled=body.attack_enabled,
        benign_persuasion=body.benign_persuasion,
        attack_payload=body.attack_payload,
        max_react_steps=body.max_react_steps,
        human_players=(body.human_seat,) if body.human_seat else (),
    )
