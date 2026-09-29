# Agent harness — measuring prompt injection *between* agents

> In a multi-agent system, A's output is B's input. Nobody treats it as
> untrusted, because "our own agents" feel trustworthy. This is an instrument
> that measures how far that assumption is wrong, and what it costs to fix.

The environment is Werewolf. Not because the game is interesting, but because
it has three properties that are hard to find together:

- **the attacker is a legal participant with a rule-given motive.** No external
  adversary has to be assumed — a werewolf is *supposed* to manipulate the vote.
- **the outcome is binary and machine-checkable.** A vote either moved to the
  attacker's target or it did not. No human grader in the main loop.
- **attack and honest persuasion are not separable by surface form.** "I think 4
  is the wolf, everyone vote 4" is either normal play or an attack depending on
  who said it — and the defence cannot see who said it.

That third property is the whole point. It means over-defense is *measurable*:
a filter aggressive enough to stop the wolves also silences honest villagers,
and the village loses anyway. Existing injection benchmarks treat the attacker
as an outsider, so they can report "we blocked 100%" without ever pricing it.

---

## What is actually here

```
engine/     deterministic referee. Contains no model call, by rule.
harness/    the contribution: ReAct loop, tools, memory, context, guard stack
attacks/    payload corpus (dev/holdout split) + a benign persuasion corpus
evalkit/    runner, six metric axes, judges, statistics
human/      a human seat, for the baseline that makes the numbers readable
server/     FastAPI + SQLite: holds the API key, runs batches, serves logs
web/        three pages; the replay view is the one that matters
tests/      338 tests, including mandatory coverage of view isolation
```

Rules are **not** hand-written. Roles, night-action priority, save/poison
conflict resolution, vote legality and win conditions come from the MIT-licensed
[`werewolf-engine`](https://pypi.org/project/werewolf-engine/) package;
`engine/` is an adapter that seeds it deterministically and adds the things it
has no concept of: a speech phase, abstention, a round cap, and a turn-based
night in which every action is an agent's own decision rather than a script.
The game is the environment, not the deliverable — writing a fifth Werewolf
implementation would have been the wrong use of the time.

---

## Run it, with no API key

Everything below works offline against a deterministic scripted client
(`harness/providers/mock.py`). Nothing needs a key until you point it at a real
gateway.

```bash
pip install -r werewolf_harness/requirements.txt

# one game, printed: roles, speeches, injections, votes, cost
python -m werewolf_harness.cli demo --seed 5

# the guard ablation, with Wilson intervals
python -m werewolf_harness.cli ablation --seeds 20

# the dashboard (replay / config / experiments)
python -m werewolf_harness.cli seed-db      # a few games to look at
python -m werewolf_harness.cli serve        # http://127.0.0.1:8000

# take a seat yourself; the other players are not told
python -m werewolf_harness.cli play --seat 3

pytest werewolf_harness/tests -q
```

---

## Claude on your subscription (`claude -p`)

```bash
claude                      # once: log in with your Claude subscription
python -m werewolf_harness.cli probe --provider claude-cli --model claude-opus-5-5
python -m werewolf_harness.cli demo  --provider claude-cli --model claude-opus-5-5 --seed 5
```

Each harness step is one `claude -p` call: tools off, one turn, the harness's
JSON protocol. Usage counts against the subscription, not an API key; the game
log records `billing: Claude subscription` and a cost of zero (token counts are
still reported). What the client pins down, each found by testing the real CLI:

- **billing**: `ANTHROPIC_API_KEY` is removed from the child's environment —
  if set, the CLI bills it instead of the subscription. `--bare` is not used
  for the same reason.
- **nothing from the machine in the prompt**: without `--setting-sources ""`
  a `CLAUDE.md` in the working directory reached the model despite
  `--system-prompt` (a planted instruction was obeyed). The child runs in an
  empty directory of its own.
- **no inherited session**: variables from a parent Claude Code session
  (including its effort level) are stripped; effort is always explicit.
- **one call per step**: the CLI's auxiliary calls are switched off
  (`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`); they doubled the calls.
- **real turn boundaries**: a multi-step turn is one CLI session continued
  with `--resume`. Flattening the history into one prompt would let a speech
  that forges "[assistant]" cross a role boundary — an injection surface
  created by the transport and measured as the model's.
- **usage limits**: hitting the subscription's cap abandons the game (counted
  as crashed) instead of letting the seat default to an abstention it never
  chose.

Not removable, and recorded under `provider_settings.injected_by_cli`: the
CLI prefixes the system prompt with an Agent-SDK identity line and adds
environment, model and date lines. They are identical for every seat and arm.

---

## Claude, directly (Anthropic API, per-token)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m werewolf_harness.cli probe --provider anthropic --model claude-opus-5-5
python -m werewolf_harness.cli demo  --provider anthropic --model claude-opus-5-5 --seed 5
```

or `setup-gateway` with the key in `.env`, then add the model on the config
page. The provider decides the client, never the model name: a relay serves
`claude-*` names too, and those stay on the relay.

Five things differ from a relay model, and each would otherwise show up as a
wrong number rather than an error. All five are recorded per game under
`config.provider_settings`:

| | what happens | why it matters here |
|---|---|---|
| **sampling** | no `temperature` is sent — current Claude models reject it | runs are not deterministic; the variance of an arm includes the model's own sampling. Report it as a limitation |
| **thinking** | always on; `effort` (default `medium`) is the control, and it is a variable of the experiment | a reply cap sized for a non-thinking model cuts the reply off mid-thought — every turn then falls through to the default action |
| **history** | thinking blocks are replayed verbatim, and a Claude turn is never trimmed (`context_policy: append_only`) | an edited history invalidates the thinking in it. The turns that run long enough to trim are the guard-blocked ones, so trimming would have degraded the guarded arms specifically |
| **refusals** | recorded as `block_reason: "refusal"`, and the turn goes straight to its default — no re-ask | a decline is data in an injection study, not a malformed reply |
| **fallbacks** | on by default; a turn answered by another model is tagged `served_by` | a seat that silently changes model corrupts the per-seat log. For the formal arms consider `--fallbacks off`, so a refusal is counted as one |

`tests/fake_gateway.py --check` runs full games through the real `anthropic`
SDK against a local server that enforces the Messages API's rules (thinking
replayed unmodified and in its own conversation, every `tool_use` answered, no
sampling parameters, no forced `tool_choice`) and records every request it
would reject — a rejected request is absorbed by the loop's recovery and
becomes an abstention, so a game that "finished" proves nothing on its own. It
includes a negative control: with the old trimming switched back on, the check
must fail, and it does.

---

## Pointing it at a real gateway

The key never touches the front end and never touches a command line. It lives
in a git-ignored `.env` (or is typed once into the config page, which stores it
server-side and returns only a mask).

```bash
cp werewolf_harness/.env.example .env    # fill in base_url and token(s)
set -a && . ./.env && set +a             # bash/zsh; fish: `export (cat .env | ...)`
python -m werewolf_harness.cli setup-gateway
```

Then add a model with its name **copied from the gateway's model list, never
typed** — relay names drift from vendor names (`claude-sonnet-4-5-20250929` vs
`claude-sonnet-4.5`), and a typo comes back as `400 no available channel`,
which reads like a broken key and is not one. Add it on the config page, or:

```bash
python -m werewolf_harness.cli probe --model <copied-name> --group <group>
```

`probe` is the phase-0 checklist, and it is not optional. It answers eight
questions that each cost a whole batch to discover halfway through: is the
model reachable, does a `tools` request come back with a real `tool_calls`
structure, do the arguments parse, can a tool result be fed back, **is
`temperature=0` actually honoured**, is usage reported at all, and how slow is
a round trip. A model that fails the tool-calling items is not rejected — it is
demoted to `tool_mode="json_prompt"` and the game log records which mode it
ran in.

The temperature item is the one to read carefully. Relays often drop the
parameter on the way upstream; if three identical requests give three different
answers, part of the variance in that arm belongs to the gateway, not the
model, and it has to be reported as a limitation rather than averaged away.

To exercise the whole HTTP path — both tool modes, message pairing, retries,
error mapping — without a network or a key:

```bash
python -m werewolf_harness.tests.fake_gateway --check
```

That is a mock of the *wire*, not of the model: a local server speaking
chat-completions, so the real client plays real games over real HTTP. It is
what to run before the first paid batch, and it is how the missing retry in the
probe was found.

---

## Results

**There are none yet, and this section will not be filled in with numbers from
the offline client.** The mock exists to exercise the instrument — its
susceptibility to injection is a constant I chose, so reading a "finding" off
it would be reading my own assumption back. Its runs are labelled `model=mock`
and excluded from every reported result.

What the instrument produces, per guard configuration, paired by seed:

| guard | hijack (speech) | hijack (tool return) | latent compromise | false-block rate | village win | tokens/game |
|---|---|---|---|---|---|---|
| none | — | — | — | 0% by construction | — | — |
| +L1 | | | | | | |
| +L1+L2 | | | | | | |
| +L1+L2+L3 | | | | | | |
| +L1+L2+L3+E | | | | | | |

The claim this table is built to support or refute is that the last two columns
move in the *opposite* direction from the first two — that pushing the hijack
rate down costs village win rate, and that the cost is large enough to matter.

Three things get reported next to it, always:

- **judge/human agreement** on the leakage axis — the ceiling on how much any
  judged number can be believed;
- **crashed games per arm** — if one configuration crashes more, that is a
  finding, not a footnote;
- **raw token counts, not billed cost** — gateways cache prompts, so a batch's
  invoice depends on the order it ran in. Token counts are comparable; money is
  not.

---

## The two attack paths

The same payload is scored separately depending on how it reached the victim:

- **path A — speech.** The agent reads it in the round it was spoken.
- **path B — tool return.** The agent calls `query_history(5, 2)` and the
  payload comes back quoted inside retrieved data.

Path B needs no extra machinery: the transcript persists, so anything said in
round *r* is retrievable in round *r+1*. It is the case worth watching, because
retrieved data tends to arrive with an implicit authority that a stranger's
speech does not — the same shape as a poisoned document in a RAG pipeline.

---

## The guard stack

| layer | what it does | what it costs |
|---|---|---|
| **L1** isolate | fences everything written by others; declares it data, not instructions | ~nothing; cannot be defeated by closing the fence early (tested) |
| **L2** filter | strips forged conversation structure and imperative directives | catches honest urgency: "you must vote 4 today" reads identically |
| **L3** verify | output-side: ballot must match the speech, and obedience without evidence is refused | its sharp end also refuses agreement — an agent that independently reached the same conclusion looks the same |
| **E** evidence | no attributing a statement to a player you did not look up | extra tool calls per turn |

L2's pattern list was written against the **dev** payload split only; a
**holdout** split exists that no pattern targets, and a test asserts the holdout
is not fully neutralised — otherwise the filter would be scored on its own
training set.

One class is deliberately, provably out of reach of L2: a payload that simply
*invents* what someone said carries no forged structure and no imperative. It is
indistinguishable in surface form from an honest player misremembering. That is
what layer E is for — the claim is checkable even though the sentence is not
classifiable.

---

## Design decisions worth defending

1. **The referee is if-else, never a model.** Vote counting, win detection and
   information visibility are fully formalisable, so they are code. A model is
   used only where judgement is genuinely ambiguous. There is a test that fails
   if anything in `engine/` so much as refers to a model.
2. **An agent turn is a bounded ReAct loop, not one call.** Otherwise there is
   no tool layer, no memory, no context budget, and no second attack channel —
   the interesting engineering has nowhere to live.
3. **No agent framework.** The loop's details — when to trim, what counts as a
   repeated call, what a blocked action does, what a failed turn falls back to —
   *are* the measurements. A framework would own all four.
4. **The context deliberately withholds earlier rounds.** Only the current round
   is in the window; recall goes through `query_history`. That is what makes the
   tools load-bearing and evidence enforcement meaningful.
5. **Nothing is pushed to 0%.** A defence that reports a perfect block rate has
   almost certainly not measured what it broke.

## Model access

One OpenAI-compatible gateway, one client, no per-vendor branching. Three
things the access layer handles because each one otherwise surfaces halfway
through a paid batch:

- **a probe** decides per model whether native function calling actually works
  end-to-end (call → arguments parse → tool result fed back). A model that fails
  drops to `tool_mode="json_prompt"`, runs the identical loop with the tools
  described in the prompt, and carries that label in every game log.
- **actionable gateway errors.** `no available channel` becomes "the token's
  group does not match this model"; each model config can point at its own
  provider, because one token is usually bound to one group.
- **the key stays server-side.** The browser creates a provider, sees
  `sk-a****3f2a`, tests it, deletes it — and can never read it back or reach the
  gateway itself.

## Further reading

- [`ENGINEERING.md`](ENGINEERING.md) — architecture, ReAct loop, tool schema,
  guard internals, failure handling, and the things that went wrong.
- [`FINDINGS.md`](FINDINGS.md) — experimental method: controls, sample-size
  planning, judge calibration, what is frozen before the batch runs, and the
  limitations that stay limitations.

Aligned with OWASP Top 10 for Agentic Applications: **ASI01** (agent goal
hijack), **ASI07** (insecure inter-agent communication), **ASI08** (cascading
failures).
