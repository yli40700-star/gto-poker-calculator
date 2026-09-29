"""Command line entry point.

    python -m werewolf_harness.cli demo                 one offline game, printed
    python -m werewolf_harness.cli ablation --seeds 20  the guard sweep + table
    python -m werewolf_harness.cli probe --model NAME   the phase-0 gateway probe
                    [--provider anthropic|claude-cli]   ...or Claude, directly
    python -m werewolf_harness.cli play                 join a game as a human
    python -m werewolf_harness.cli serve                the dashboard backend

Everything except `probe` and a non-mock `--model` runs with no API key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .evalkit import metrics, stats
from .evalkit.runner import RunConfig, run_batch, run_game

GUARD_ARMS = [
    ((), False, "none"),
    (("L1",), False, "L1"),
    (("L1", "L2"), False, "L1+L2"),
    (("L1", "L2", "L3"), False, "L1+L2+L3"),
    (("L1", "L2", "L3"), True, "L1+L2+L3+E"),
]


def cmd_demo(args) -> int:
    cfg = RunConfig(
        seed=args.seed,
        model=_model_config(args),
        guard_layers=tuple(args.guard),
        evidence_forced=args.evidence,
        attack_enabled=not args.no_attack,
        trace_dir=args.trace_dir,
    )
    log = run_game(cfg)
    _print_game(log)
    if args.out:
        Path(args.out).write_text(json.dumps(log, ensure_ascii=False, indent=2), "utf-8")
        print(f"\nlog written to {args.out}")
    return 1 if log["outcome"]["crashed"] else 0


def cmd_ablation(args) -> int:
    arms = GUARD_ARMS
    seeds_count, workers = args.seeds, args.workers
    if args.config:
        spec = json.loads(Path(args.config).read_text("utf-8"))
        seeds_count = spec.get("seeds", seeds_count)
        workers = spec.get("workers", workers)
        arms = [
            (tuple(a.get("guard_layers", [])), bool(a.get("evidence_forced")),
             ("+".join(a.get("guard_layers", [])) or "none")
             + ("+E" if a.get("evidence_forced") else ""))
            for a in spec.get("arms", [])
        ] or GUARD_ARMS
    seeds = list(range(seeds_count))
    model = _model_config(args)
    configs = []
    for layers, evidence, _label in arms:
        for seed in seeds:
            common = dict(
                seed=seed, model=model, guard_layers=layers, evidence_forced=evidence,
                max_react_steps=args.max_steps,
            )
            configs.append(RunConfig(attack_enabled=True, **common))
            configs.append(RunConfig(benign_persuasion=True, **common))

    done = [0]

    def progress(n, total, _log):
        done[0] = n
        if n % 10 == 0 or n == total:
            print(f"  {n}/{total} games", file=sys.stderr)

    print(f"running {len(configs)} games ({len(arms)} arms x {len(seeds)} seeds "
          f"x 2 conditions), model={model.get('model_name')}", file=sys.stderr)
    logs = run_batch(configs, workers=workers, out_dir=args.out_dir,
                     on_progress=progress)

    rows = []
    for layers, evidence, label in arms:
        arm = [
            log for log in logs
            if log["config"]["guard_layers"] == list(layers)
            and log["config"]["evidence_forced"] == evidence
        ]
        attack_logs = [log for log in arm if log["config"]["attack_enabled"]]
        benign_logs = [log for log in arm if log["config"]["benign_persuasion"]]
        summary = metrics.summarise(attack_logs)
        over = metrics.overdefense(benign_logs)

        trials = [t for log in attack_logs for t in metrics.injection_trials(log)]
        speech = [t for t in trials if t["channel"] == "speech" and t["task"] == "vote"]
        tool = [t for t in trials if t["channel"] == "tool_return" and t["task"] == "vote"]
        rows.append(
            {
                "guard": label,
                "hijack_speech": stats.wilson(sum(t["hijacked"] for t in speech), len(speech)),
                "hijack_tool": stats.wilson(sum(t["hijacked"] for t in tool), len(tool)),
                "latent": stats.wilson(
                    sum(t["latent_compromise"] for t in trials), len(trials)
                ),
                "false_block": over["false_block_rate"],
                "village_win": summary["village_win_rate"],
                "tokens": summary["mean_tokens_per_game"],
                "crashed": summary["crashed"],
            }
        )

    _print_table(rows, model.get("model_name", "?"), len(seeds))
    if args.out_dir:
        out = Path(args.out_dir) / "ablation_summary.json"
        out.write_text(
            json.dumps(
                [
                    {k: (v.to_dict() if isinstance(v, stats.Interval) else v)
                     for k, v in row.items()}
                    for row in rows
                ],
                indent=2,
            ),
            "utf-8",
        )
        print(f"\nsummary written to {out}")
    return 0


def cmd_seed_db(args) -> int:
    """Fill the dashboard's database with offline games so the replay page has
    something to show on a fresh clone."""
    from .server import db as dbmod

    conn = dbmod.connect(args.db)
    made = 0
    for seed in range(args.games):
        for layers, evidence, _label in (GUARD_ARMS[0], GUARD_ARMS[2], GUARD_ARMS[3]):
            for benign in (False, True):
                cfg = RunConfig(
                    seed=seed,
                    guard_layers=layers,
                    evidence_forced=evidence,
                    attack_enabled=not benign,
                    benign_persuasion=benign,
                )
                log = run_game(cfg)
                dbmod.save_game(conn, log, label=cfg.label())
                made += 1
    print(f"wrote {made} offline games to {args.db}")
    return 0


def cmd_setup_gateway(args) -> int:
    """Register relay providers from the environment.

    Keys are read from the environment (or a git-ignored .env), never passed on
    the command line, so they do not end up in shell history. A gateway usually
    binds one token to one channel group, so each group becomes its own
    provider and a model points at the one that can actually serve it.
    """
    from .server import db as dbmod

    # (name, base_url, key, kind) for every provider the environment describes.
    wanted: list[tuple[str, str, str, str]] = []

    base = os.getenv("LLM_BASE_URL", "")
    if base:
        if not base.rstrip("/").endswith("/v1"):
            print(f"warning: {base} does not end in /v1, which most gateways require",
                  file=sys.stderr)
        groups = [
            (name, os.getenv(var, ""))
            for name, var in (
                ("openai-group", "LLM_KEY_OPENAI_GROUP"),
                ("claude-group", "LLM_KEY_CLAUDE_GROUP"),
            )
        ]
        groups = [(n, k) for n, k in groups if k]
        if not groups and os.getenv("LLM_API_KEY"):
            groups = [("default", os.environ["LLM_API_KEY"])]
        wanted += [(n, base, k, "openai_compat") for n, k in groups]

    if os.getenv("ANTHROPIC_API_KEY"):
        wanted.append(("anthropic",
                       os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com",
                       os.environ["ANTHROPIC_API_KEY"], "anthropic"))

    if not wanted:
        print("no keys in the environment: set ANTHROPIC_API_KEY for Claude direct, "
              "or LLM_BASE_URL with LLM_KEY_*_GROUP / LLM_API_KEY for a relay "
              "(see .env.example)", file=sys.stderr)
        return 2

    conn = dbmod.connect(args.db)
    existing = {p["name"] for p in dbmod.list_providers(conn)}
    for name, url, key, kind in wanted:
        if name in existing:
            print(f"  {name}: already registered, left alone")
            continue
        provider = dbmod.add_provider(conn, name, url, key, kind)
        print(f"  {name}: {provider['api_key_masked']} -> {url}  ({kind})")

    print(f"\n{len(dbmod.list_providers(conn))} provider(s) in {args.db}")
    print("Next: add a model on the config page, then probe it before any batch:")
    print("  python -m werewolf_harness.cli serve")
    if any(kind == "anthropic" for *_, kind in wanted):
        print("  python -m werewolf_harness.cli probe --provider anthropic --model claude-opus-5-5")
    if any(kind == "openai_compat" for *_, kind in wanted):
        print("  python -m werewolf_harness.cli probe --model <name COPIED from the relay's "
              "model list> --group <group>")
    return 0


def cmd_probe(args) -> int:
    from .harness.providers import ProviderError, build_client, probe_model

    config = _model_config(args)
    if not config.get("api_key") and args.provider != "claude-cli":
        var = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "LLM_API_KEY"
        print(f"no API key: pass --api-key or set {var}", file=sys.stderr)
        return 2
    try:
        client = build_client(config)
    except ProviderError as exc:
        print(f"{exc}\n  hint: {exc.hint}", file=sys.stderr)
        return 2
    try:
        result = probe_model(client, check_temperature=not args.skip_temperature)
    finally:
        close = getattr(client, "close", None)
        if close:
            close()
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.reachable else 1


def cmd_play(args) -> int:
    from .human.cli import play

    return play(seed=args.seed, seat=args.seat, guard=tuple(args.guard),
                attack=not args.no_attack)


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed: pip install -r requirements.txt", file=sys.stderr)
        return 2
    uvicorn.run("werewolf_harness.server.app:app", host=args.host, port=args.port,
                reload=args.reload)
    return 0


# ----------------------------------------------------------------- output

def _model_config(args) -> dict:
    if args.model == "mock":
        return {"model_name": "mock"}
    if getattr(args, "provider", "relay") == "claude-cli":
        return {
            "provider": "claude_cli",
            "model_name": args.model,
            "display_name": args.model,
            "effort": getattr(args, "effort", "medium"),
            "tool_mode": "json_prompt",
        }
    if getattr(args, "provider", "relay") == "anthropic":
        return {
            "provider": "anthropic",
            "model_name": args.model,
            "display_name": args.model,
            "api_key": getattr(args, "api_key", None) or os.getenv("ANTHROPIC_API_KEY", ""),
            "base_url": getattr(args, "base_url", None) or os.getenv("ANTHROPIC_BASE_URL"),
            "effort": getattr(args, "effort", "medium"),
            "fallbacks": getattr(args, "fallbacks", "default"),
        }
    return {
        "model_name": args.model,
        "display_name": args.model,
        "api_key": getattr(args, "api_key", None) or os.getenv("LLM_API_KEY", ""),
        "base_url": getattr(args, "base_url", None) or os.getenv("LLM_BASE_URL", ""),
        "tool_mode": getattr(args, "tool_mode", "native"),
        "group": getattr(args, "group", None),
    }


def _print_game(log: dict) -> None:
    gt = log["ground_truth"]
    print(f"game {log['game_id']}  seed={log['seed']}  "
          f"guard={'+'.join(log['config']['guard_layers']) or 'none'}"
          f"{'+E' if log['config']['evidence_forced'] else ''}  "
          f"model={log['config']['model']}")
    print(f"roles: {gt['roles']}")
    for rnd in log["rounds"]:
        print(f"\n--- round {rnd['round']} (alive {rnd['alive']}, "
              f"night deaths {rnd['night_deaths']})")
        for payload in rnd["injected_payloads"]:
            print(f"    [inject] {payload['payload_id']} by p{payload['attacker']} "
                  f"-> target p{payload['target']}")
        for agent in rnd["agents"]:
            if agent["speech"]:
                # Printed whole, wrapped. A fixed-width cut used to drop the
                # tail -- which is exactly where an injected payload is
                # appended, so the one command a new reader runs first was
                # hiding the thing this project is about.
                _print_speech(agent["player_id"], agent["speech"])
            if agent["task"] == "vote":
                blocked = f"  [{len(agent['guard_blocks'])} blocked]" if agent["guard_blocks"] else ""
                print(f"  p{agent['player_id']} votes {agent['vote']}"
                      f" ({len(agent['react_trace'])} steps){blocked}")
        print(f"  tally {rnd['vote_counts']} -> exiled {rnd['exiled']}")
    out = log["outcome"]
    print(f"\nwinner: {out['winner']}  tokens: {out['total_prompt_tokens']}+"
          f"{out['total_completion_tokens']}  cost: ${out['total_cost_usd']:.4f}  "
          f"{out['total_duration_s']}s")


def _print_speech(player_id: int, speech: str) -> None:
    import textwrap

    head = f"  p{player_id}: "
    for i, line in enumerate(textwrap.wrap(speech, width=96) or [""]):
        print(f"{head}{line}" if i == 0 else f"{' ' * len(head)}{line}")


def _print_table(rows: list[dict], model: str, seeds: int) -> None:
    print(f"\n{'':-<96}")
    print(f"guard ablation -- model={model}, {seeds} seeds per arm, paired by seed")
    print(f"{'':-<96}")
    print(f"{'guard':<12} {'hijack(speech)':>22} {'hijack(tool)':>22} "
          f"{'false-block':>12} {'village-win':>12} {'tokens':>8}")
    for row in rows:
        print(
            f"{row['guard']:<12} "
            f"{str(row['hijack_speech']):>22} "
            f"{str(row['hijack_tool']):>22} "
            f"{_pct(row['false_block']):>12} "
            f"{_pct(row['village_win']):>12} "
            f"{row['tokens'] or 0:>8.0f}"
        )
    print(f"{'':-<96}")
    print("intervals are Wilson 95%. Overlapping intervals mean no detectable "
          "difference, not equality.")


def _pct(value) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="werewolf_harness")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--model", default="mock",
                       help="gateway model name, or 'mock' for the offline client")
        p.add_argument("--api-key", default=None)
        p.add_argument("--base-url", default=None)
        p.add_argument("--tool-mode", default="native", choices=["native", "json_prompt"])
        p.add_argument("--group", default=None, help="gateway token group")
        p.add_argument("--provider", default="relay",
                       choices=["relay", "anthropic", "claude-cli"],
                       help="an OpenAI-compatible relay; Claude via the Anthropic API "
                            "(per-token billing); or Claude via `claude -p` (your "
                            "Claude subscription)")
        p.add_argument("--effort", default="medium",
                       choices=["low", "medium", "high", "xhigh", "max"],
                       help="Claude thinking effort (recorded in every game log)")
        p.add_argument("--fallbacks", default="default", choices=["default", "off"],
                       help="Claude refusal fallbacks; 'off' records a refusal as a refusal")

    demo = sub.add_parser("demo", help="play one game and print it")
    add_common(demo)
    demo.add_argument("--seed", type=int, default=1)
    demo.add_argument("--guard", nargs="*", default=["L1", "L2"], choices=["L1", "L2", "L3"])
    demo.add_argument("--evidence", action="store_true")
    demo.add_argument("--no-attack", action="store_true")
    demo.add_argument("--trace-dir", default=None)
    demo.add_argument("--out", default=None)
    demo.set_defaults(func=cmd_demo)

    abl = sub.add_parser("ablation", help="run the guard sweep and print the table")
    add_common(abl)
    abl.add_argument("--seeds", type=int, default=10)
    abl.add_argument("--workers", type=int, default=3)
    abl.add_argument("--max-steps", type=int, default=8)
    abl.add_argument("--out-dir", default=None)
    abl.add_argument("--config", default=None,
                     help="JSON experiment file; see configs/experiment.example.json")
    abl.set_defaults(func=cmd_ablation)

    seed_db = sub.add_parser("seed-db", help="fill the dashboard db with offline games")
    seed_db.add_argument("--db", default="werewolf_harness.db")
    seed_db.add_argument("--games", type=int, default=3, help="seeds per configuration")
    seed_db.set_defaults(func=cmd_seed_db)

    setup = sub.add_parser("setup-gateway",
                           help="register providers (relay and/or Anthropic) from the environment")
    setup.add_argument("--db", default="werewolf_harness.db")
    setup.set_defaults(func=cmd_setup_gateway)

    probe = sub.add_parser("probe", help="phase-0 gateway checks for one model")
    add_common(probe)
    probe.add_argument("--skip-temperature", action="store_true")
    probe.set_defaults(func=cmd_probe)

    play = sub.add_parser("play", help="take a seat at the table yourself")
    play.add_argument("--seed", type=int, default=1)
    play.add_argument("--seat", type=int, default=1, choices=range(1, 9))
    play.add_argument("--guard", nargs="*", default=["L1", "L2"], choices=["L1", "L2", "L3"])
    play.add_argument("--no-attack", action="store_true")
    play.set_defaults(func=cmd_play)

    serve = sub.add_parser("serve", help="run the dashboard backend")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
