"""Output contracts.

Everything crossing the model boundary is validated here before it can touch
the world. Validation is hand-written rather than pulled from jsonschema: the
shapes are tiny, and the error strings are fed back to the model on a retry, so
they have to be actionable ("target_id must be an integer", not "does not
match schema").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class SchemaError(Exception):
    """Raised when a model produced something that does not fit the contract."""


# ---------------------------------------------------------------- belief

@dataclass
class BeliefEntry:
    player_id: int
    suspicion: float  # 0 = certainly village, 1 = certainly wolf
    guessed_role: str | None = None
    reason: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    updated_at_round: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "BeliefEntry":
        return BeliefEntry(
            player_id=int(d["player_id"]),
            suspicion=float(d["suspicion"]),
            guessed_role=d.get("guessed_role"),
            reason=d.get("reason", ""),
            evidence_refs=list(d.get("evidence_refs", [])),
            updated_at_round=int(d.get("updated_at_round", 0)),
        )


# ------------------------------------------------------------ react step

@dataclass
class ReActStep:
    step: int
    thought: str = ""
    action: str = ""
    args: dict = field(default_factory=dict)
    observation: str = ""
    tokens: int = 0
    latency_ms: int = 0
    guard_blocked: bool = False
    block_reason: str | None = None
    guard_detections: list[dict] = field(default_factory=list)
    injected: bool = False  # an attack payload was present in this observation
    # Set only when a fallback model answered instead of the configured one.
    served_by: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------- validation

_TERMINAL = {"speak", "vote", "night_check", "night_kill",
             "night_save", "night_poison", "night_skip"}


def validate_call(name: Any, args: Any, registry) -> tuple[str, dict]:
    """Validate a tool call against the registry.

    Three gates, in order: the name must be a registered tool (this is where
    hallucinated tools are stopped), the arguments must be an object, and each
    declared parameter must typecheck.
    """
    if not isinstance(name, str) or not name:
        raise SchemaError("action name must be a non-empty string")
    tool = registry.get(name)
    if tool is None:
        known = ", ".join(sorted(registry.names()))
        raise SchemaError(
            f"unknown tool {name!r}; you may only call one of: {known}"
        )
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise SchemaError(f"arguments for {name} must be a JSON object")

    cleaned: dict = {}
    for pname, spec in tool.params.items():
        required = spec.get("required", True)
        if pname not in args or args[pname] is None:
            if required:
                raise SchemaError(f"{name} is missing required argument {pname!r}")
            cleaned[pname] = spec.get("default")
            continue
        cleaned[pname] = _coerce(name, pname, args[pname], spec)

    extra = set(args) - set(tool.params)
    if extra:
        raise SchemaError(
            f"{name} does not take argument(s): {', '.join(sorted(extra))}"
        )
    return name, cleaned


def _coerce(tool: str, pname: str, value: Any, spec: dict) -> Any:
    kind = spec["type"]
    if kind == "int":
        if isinstance(value, bool):
            raise SchemaError(f"{tool}.{pname} must be an integer")
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value.strip())
        if not isinstance(value, int):
            raise SchemaError(f"{tool}.{pname} must be an integer, got {value!r}")
        return value
    if kind == "float":
        if isinstance(value, bool):
            raise SchemaError(f"{tool}.{pname} must be a number")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise SchemaError(f"{tool}.{pname} must be a number, got {value!r}")
        lo, hi = spec.get("range", (None, None))
        if lo is not None and not (lo <= value <= hi):
            raise SchemaError(f"{tool}.{pname} must be between {lo} and {hi}")
        return value
    if kind == "str":
        if not isinstance(value, str):
            raise SchemaError(f"{tool}.{pname} must be a string")
        max_len = spec.get("max_len")
        if max_len and len(value) > max_len:
            value = value[:max_len]
        if spec.get("non_empty") and not value.strip():
            raise SchemaError(f"{tool}.{pname} must not be empty")
        return value
    if kind == "list[str]":
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            raise SchemaError(f"{tool}.{pname} must be a list of strings")
        return value
    raise SchemaError(f"internal: unknown parameter type {kind!r}")


def is_terminal(name: str) -> bool:
    return name in _TERMINAL


# ------------------------------------------------------------- game log

def new_game_log(game_id: str, seed: int, config: dict) -> dict:
    """The log structure frozen in the design doc.

    Every metric axis reads from this and nothing else, so a missing field
    means a re-run of the whole batch. Do not remove fields; add only.
    """
    return {
        "game_id": game_id,
        "seed": seed,
        "config": {
            "model": config.get("model"),
            "guard_layers": list(config.get("guard_layers", [])),
            "evidence_forced": bool(config.get("evidence_forced", False)),
            "max_react_steps": int(config.get("max_react_steps", 8)),
            "attack_enabled": bool(config.get("attack_enabled", False)),
            "attack_type": config.get("attack_type"),
            "tool_mode": config.get("tool_mode", "native"),
            "temperature": config.get("temperature", 0.7),
            "benign_persuasion": bool(config.get("benign_persuasion", False)),
        },
        "ground_truth": {"roles": {}, "winner": None},
        "rounds": [],
        "outcome": {
            "winner": None,
            "crashed": False,
            "crash_reason": None,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost_usd": 0.0,
            "total_duration_s": 0.0,
        },
    }
