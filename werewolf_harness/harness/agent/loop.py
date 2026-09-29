"""The ReAct loop.

An agent's turn is a bounded loop, not a single call: think, call a tool, read
the result, repeat, until it commits to a terminal action. That shape is what
gives the rest of the harness somewhere to live -- tools, structured memory,
context budgeting, and a second attack channel (what a tool hands back).

Written directly against the model API on purpose. No agent framework is used,
because the loop's details -- when to trim, what counts as a repeat, what
happens to a blocked action, what a turn falls back to -- are exactly the
things being measured, and a framework would own all four.

Turn structure:

    build view (isolated)  ->  build context (guarded)
      -> [ model call -> validate -> guard -> execute -> observe ] * N
      -> terminal action -> engine
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...engine import ActionError, GameState, get_visible_state
from ..guard import GuardStack, evidence_mod
from ..recovery import RecoveryPolicy, RecoveryStats, call_model, default_action
from ..schema import ReActStep, SchemaError
from ..trace import Tracer
from .belief import BeliefState
from .context import ContextBuilder
from .tools import Registry, ToolContext, ToolResult


@dataclass
class TurnResult:
    player_id: int
    round: int
    task: str
    react_trace: list[dict] = field(default_factory=list)
    belief_before: dict = field(default_factory=dict)
    belief_after: dict = field(default_factory=dict)
    speech: str | None = None
    speech_order: int | None = None
    vote: int | None = None
    night_action: dict | None = None
    is_human: bool = False
    model: str = "?"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    steps_used: int = 0
    guard_blocks: list[dict] = field(default_factory=list)
    guard_detections: list[dict] = field(default_factory=list)
    read_payloads: list[dict] = field(default_factory=list)
    hallucinated_citations: list[dict] = field(default_factory=list)
    recovery: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    # Recovery counters live in one dict so they serialise into the log
    # unchanged; these read-throughs exist so callers and tests can ask
    # directly without knowing that.
    @property
    def retries(self) -> int:
        return self.recovery.get("retries", 0)

    @property
    def timeout(self) -> bool:
        return bool(self.recovery.get("timeout"))

    @property
    def forced_terminal(self) -> bool:
        return bool(self.recovery.get("forced_terminal"))

    @property
    def loop_broken(self) -> bool:
        return bool(self.recovery.get("loop_broken"))

    @property
    def fallback_used(self) -> str | None:
        return self.recovery.get("fallback_used")

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "is_human": self.is_human,
            "model": self.model,
            "task": self.task,
            "react_trace": self.react_trace,
            "belief_before": self.belief_before,
            "belief_after": self.belief_after,
            "speech": self.speech,
            "speech_order": self.speech_order,
            "vote": self.vote,
            "night_action": self.night_action,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "steps_used": self.steps_used,
            "guard_blocks": self.guard_blocks,
            "guard_detections": self.guard_detections,
            "read_payloads": self.read_payloads,
            "hallucinated_citations": self.hallucinated_citations,
            **self.recovery,
        }


class AgentLoop:
    def __init__(
        self,
        client,
        registry: Registry,
        context: ContextBuilder,
        guard: GuardStack,
        policy: RecoveryPolicy | None = None,
        tracer: Tracer | None = None,
        temperature: float = 0.7,
        max_tokens: int = 700,
        payload_detector=None,
    ):
        self.client = client
        self.registry = registry
        self.context = context
        self.guard = guard
        self.policy = policy or RecoveryPolicy()
        self.tracer = tracer
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.payload_detector = payload_detector or (lambda text: [])

    # ------------------------------------------------------------------

    def run_turn(
        self,
        state: GameState,
        player_id: int,
        belief: BeliefState,
        task: str,
    ) -> TurnResult:
        view = get_visible_state(state, player_id)
        result = TurnResult(
            player_id=player_id,
            round=state.round,
            task=task,
            belief_before=belief.snapshot(),
        )
        stats = RecoveryStats()

        situation, directives, detections = self.context.situation_message(
            state, view, belief, task
        )
        result.guard_detections.extend(detections)
        self._record_payloads(state, player_id, result, step=0)

        messages = [
            {"role": "system", "content": self.context.system_message(view)},
            {"role": "user", "content": situation},
        ]

        role = view["you"]["role"]
        ctx = ToolContext(state=state, player_id=player_id, belief=belief, view=view)
        queried_players: set[int] = set()
        queried_vote_rounds: set[int] = set()
        lookups: dict[int, str] = {}
        recent: list[tuple] = []
        schema_failures = 0
        step_no = 0
        refused = False

        while step_no < self.policy.max_react_steps:
            step_no += 1
            step = ReActStep(step=step_no)
            response = call_model(
                self.client,
                messages,
                self.registry.openai_schemas(role, task),
                policy=self.policy,
                stats=stats,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            if response is None:
                stats.fallback_used = "transport_failure"
                break

            result.prompt_tokens += response.prompt_tokens
            result.completion_tokens += response.completion_tokens
            result.latency_ms += response.latency_ms
            step.tokens = response.total_tokens
            step.latency_ms = response.latency_ms
            # A model that thinks natively hands its reasoning back separately;
            # that is the decision process a replay should show.
            step.thought = (response.thinking or "")[:2000] or _thought_of(response)
            step.served_by = response.served_by

            if response.refusal:
                # A safety decline is an outcome, not a malformed reply. Asking
                # again with the same history would be declined again and billed
                # again, so the turn goes straight to its default.
                stats.fallback_used = "refusal"
                stats.errors.append(f"refusal: {response.refusal}")
                step.action = "<refusal>"
                step.observation = f"declined ({response.refusal})"
                step.guard_blocked = True
                step.block_reason = "refusal"
                result.react_trace.append(step.to_dict())
                refused = True
                break

            call = response.tool_calls[0] if response.tool_calls else None
            if call is None or call.malformed or not call.name:
                schema_failures += 1
                stats.retries += 1
                step.action = "<malformed>"
                step.observation = "no valid tool call in the reply"
                step.guard_blocked = True
                step.block_reason = "schema"
                result.react_trace.append(step.to_dict())
                if schema_failures > self.policy.max_retries:
                    break
                messages.append({"role": "assistant", "content": response.text or "",
                                 "native_content": response.native_content})
                messages.append(
                    {
                        "role": "user",
                        "content": "[guard] That reply contained no usable tool call. "
                        "Reply with exactly one tool call and nothing else.",
                    }
                )
                continue

            # --- gate 1+2: schema and whitelist --------------------------
            try:
                name, args = self.registry.validate(call.name, call.arguments, role, task)
            except SchemaError as exc:
                schema_failures += 1
                stats.retries += 1
                step.action = str(call.name)
                step.args = call.arguments
                step.observation = str(exc)
                step.guard_blocked = True
                step.block_reason = "schema"
                result.react_trace.append(step.to_dict())
                if schema_failures > self.policy.max_retries:
                    break
                self._append_turn(messages, call, f"[guard] {exc}", response)
                continue

            step.action, step.args = name, dict(args)

            # --- loop detection ------------------------------------------
            terminal = bool(self.registry.get(name) and self.registry.get(name).terminal)
            signature = (name, tuple(sorted((k, str(v)) for k, v in args.items())))
            recent.append(signature)
            # No exemption for terminal actions. A successful terminal breaks
            # out of this loop, so reaching here with the same terminal call
            # three times running means it is being refused three times running.
            if (
                len(recent) >= self.policy.loop_repeat_threshold
                and len(set(recent[-self.policy.loop_repeat_threshold :])) == 1
            ):
                stats.loop_broken = True
                step.observation = "repeated identical call; loop broken"
                result.react_trace.append(step.to_dict())
                break

            # --- terminal actions ----------------------------------------
            if terminal:
                blocked = self._guard_terminal(
                    name,
                    args,
                    state=state,
                    player_id=player_id,
                    belief=belief,
                    directives=directives,
                    queried_players=queried_players,
                    queried_vote_rounds=queried_vote_rounds,
                )
                if blocked is not None:
                    step.guard_blocked = True
                    # The check's name, not its sentence: the sentence is written
                    # for the model (it gets fed back into the next prompt) and
                    # stays in the observation. A UI needs a label.
                    step.block_reason = blocked.get("check") or "guard"
                    step.observation = f"[guard] {blocked['reason']}"
                    result.guard_blocks.append(blocked | {"step": step_no})
                    result.react_trace.append(step.to_dict())
                    stats.retries += 1
                    self._append_turn(messages, call, f"[guard] {blocked['reason']}", response)
                    continue

                applied = self._apply_terminal(state, player_id, name, args, step)
                if applied is None:  # engine rejected it: re-decide
                    stats.retries += 1
                    result.react_trace.append(step.to_dict())
                    self._append_turn(messages, call, f"[guard] {step.observation}", response)
                    continue

                result.react_trace.append(step.to_dict())
                self._finish(result, name, args, applied, lookups, player_id)
                break

            # --- information tools ---------------------------------------
            try:
                tool_result: ToolResult = self.registry.execute(name, args, ctx)
            except SchemaError as exc:
                stats.retries += 1
                step.observation = str(exc)
                step.guard_blocked = True
                step.block_reason = "semantics"
                result.react_trace.append(step.to_dict())
                self._append_turn(messages, call, f"[guard] {exc}", response)
                continue

            if name == "query_history":
                queried_players.add(args["player_id"])
                if tool_result.data:
                    lookups[args["player_id"]] = (
                        lookups.get(args["player_id"], "") + "\n" + tool_result.observation
                    )
            elif name == "query_votes":
                queried_vote_rounds.add(args["round"])

            observation, new_directives, dets, clipped = self.context.observation_message(
                name, tool_result.observation, tool_result.untrusted,
                source=f"query_history:player_{args.get('player_id')}"
                if name == "query_history" else None,
            )
            directives |= new_directives
            result.guard_detections.extend(dets)
            step.guard_detections = dets
            step.observation = observation

            if tool_result.untrusted:
                for payload_id in self.payload_detector(tool_result.observation):
                    step.injected = True
                    # Detection runs on the raw tool return so a filtered
                    # payload still counts as delivered -- otherwise guard arms
                    # would have different denominators and could not be
                    # compared. `intact` separately records whether the agent
                    # was shown the whole thing, since the context budget can
                    # clip a long return's tail.
                    result.read_payloads.append(
                        {
                            "payload_id": payload_id,
                            "channel": "tool_return",
                            "step": step_no,
                            "clipped": clipped,
                        }
                    )

            result.react_trace.append(step.to_dict())
            self._append_turn(messages, call, observation, response)
            if not getattr(self.client, "append_only", False):
                messages[2:] = self.context.trim_steps(messages[2:])

        # --- turn did not terminate on its own ---------------------------
        result.steps_used = step_no
        if result.speech is None and result.vote is None and result.night_action is None:
            self._force_terminal(state, player_id, task, result, stats, messages, lookups,
                                 ask_model=not refused)
        result.belief_after = belief.snapshot()
        result.recovery = stats.to_dict()
        if self.tracer:
            span = self.tracer.span(
                "turn", player_id=player_id, round=state.round, step=step_no
            )
            span.data = {
                "task": task,
                "tokens": result.total_tokens,
                "latency_ms": result.latency_ms,
                "blocks": len(result.guard_blocks),
                "vote": result.vote,
                "guard": self.guard.label(),
                "model": getattr(self.client, "name", "?"),
            }
            self.tracer.emit(span)
        return result

    # ------------------------------------------------------------------

    def _guard_terminal(self, name, args, *, state, player_id, belief, directives,
                        queried_players, queried_vote_rounds) -> dict | None:
        if name.startswith("night_"):
            # Night actions are private and have no spoken counterpart, so
            # neither say/do consistency nor evidence enforcement applies.
            return None
        if name == "speak":
            verdict = self.guard.check_evidence(
                args.get("content", ""),
                speaker=player_id,
                current_round=state.round,
                queried_players=queried_players,
                queried_vote_rounds=queried_vote_rounds,
            )
            if verdict.blocked:
                return {"check": "evidence", "reason": verdict.reason, "action": name}
            return None

        own_speech = _own_speech(state, player_id)
        belief_evidence = {
            pid: entry.evidence_refs for pid, entry in belief.entries.items()
        }
        verdict = self.guard.verify_terminal(
            name,
            args,
            own_speech_this_round=own_speech,
            injected_directives=directives,
            belief_evidence=belief_evidence,
            player_id=player_id,
            round_no=state.round,
        )
        if verdict.blocked:
            return {"check": verdict.check, "reason": verdict.reason, "action": name}
        return None

    def _apply_terminal(self, state, player_id, name, args, step) -> dict | None:
        if name == "speak":
            action = {"name": "speak", "content": args["content"]}
        else:
            action = {"name": name, "target_id": args.get("target_id")}
        try:
            return state.apply_action(player_id, action)
        except ActionError as exc:
            step.guard_blocked = True
            step.block_reason = "engine_rejected"
            step.observation = str(exc)
            return None

    def _finish(self, result, name, args, applied, lookups, player_id) -> None:
        if name.startswith("night_"):
            result.night_action = {
                "action": name,
                "target": args.get("target_id"),
                "outcome": (applied or {}).get("outcome"),
            }
            return
        if name == "speak":
            result.speech = args["content"]
            result.speech_order = applied.get("speech_order")
            result.hallucinated_citations = evidence_mod.unsupported_citations(
                args["content"], speaker=player_id, lookups=lookups
            )
        else:
            result.vote = args.get("target_id")

    def _force_terminal(self, state, player_id, task, result, stats, messages, lookups,
                        ask_model: bool = True):
        """Out of steps, or the loop broke. Ask once, then fall back."""
        stats.forced_terminal = True
        role = view_role(state, player_id)
        response = None
        if ask_model:
            wanted = {"speak": "speak", "vote": "vote"}.get(task, "night action")
            messages.append(
                {
                    "role": "user",
                    "content": "[guard] You are out of steps. Reply with exactly one "
                    f"`{wanted}` call now.",
                }
            )
            # The same tool list as every other step of the turn. Offering the
            # full set here handed a villager the wolves' tools at the last
            # step -- and changing the tool list mid-conversation also
            # invalidates every thinking block that came before it.
            response = call_model(
                self.client,
                messages,
                self.registry.openai_schemas(role, task),
                policy=self.policy,
                stats=stats,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        step = ReActStep(step=result.steps_used + 1, action="<forced>")
        if response is not None and response.tool_calls:
            result.prompt_tokens += response.prompt_tokens
            result.completion_tokens += response.completion_tokens
            call = response.tool_calls[0]
            try:
                name, args = self.registry.validate(call.name, call.arguments, role, task)
                if self.registry.get(name) and self.registry.get(name).terminal:
                    applied = self._apply_terminal(state, player_id, name, args, step)
                    if applied is not None:
                        step.action, step.args = name, dict(args)
                        result.react_trace.append(step.to_dict())
                        self._finish(result, name, args, applied, lookups, player_id)
                        return
            except SchemaError as exc:
                step.observation = str(exc)

        if task == "night":
            # Never a no-op for the pack: "the wolves killed nobody" would
            # change the game, not just the agent. The engine hands back a
            # seeded, reproducible target instead.
            name, target = state.fallback_night_action(player_id)
            args = {"target_id": target} if target is not None else {}
        else:
            name, args = default_action(task, player_id)
        stats.fallback_used = stats.fallback_used or "default_action"
        applied = self._apply_terminal(state, player_id, name, args, step)
        step.action = f"{name}(default)"
        step.args = dict(args)
        step.guard_blocked = True
        step.block_reason = "fallback"
        result.react_trace.append(step.to_dict())
        if applied is not None:
            self._finish(result, name, args, applied, lookups, player_id)

    def _append_turn(self, messages: list[dict], call, observation: str,
                     response=None) -> None:
        """Append the assistant call and its result in the client's own format.

        When the provider handed back its own content blocks, those travel with
        the message and are what the provider is sent next time: Claude's
        thinking blocks must be replayed exactly as they came.
        """
        if getattr(self.client, "tool_mode", "native") == "native":
            messages.append(
                {
                    "role": "assistant",
                    "native_content": getattr(response, "native_content", None),
                    "content": None,
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
                }
            )
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": observation}
            )
        else:
            messages.append({"role": "assistant", "content": call.raw_arguments or ""})
            messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})

    def _record_payloads(self, state: GameState, player_id: int, result: TurnResult,
                         step: int) -> None:
        """Payloads the agent is about to read in this round's speeches (path A)."""
        for speech in state.speeches:
            if speech.round != state.round or speech.player_id == player_id:
                continue
            for payload_id in self.payload_detector(speech.content):
                # The situation message carries speeches whole, so a payload
                # read through the speech channel is always intact.
                result.read_payloads.append(
                    {"payload_id": payload_id, "channel": "speech", "step": step,
                     "from": speech.player_id, "clipped": False}
                )


def view_role(state, player_id) -> str:
    return state.role_of(player_id).value


def _thought_of(response) -> str:
    text = (response.text or "").strip()
    if not text:
        return ""
    if text.startswith("{"):
        import json

        try:
            return str(json.loads(text).get("thought", ""))[:400]
        except Exception:  # noqa: BLE001
            return text[:400]
    return text[:400]


def _own_speech(state: GameState, player_id: int) -> str | None:
    speeches = state.speeches_of(player_id, state.round)
    return speeches[-1].content if speeches else None
