/* Replay, config and experiment pages.
 *
 * No framework and no build step: this is an instrument panel that has to open
 * from a clone with nothing installed. The replay view is the part that earns
 * its keep -- it shows where a payload entered a turn and what happened to the
 * agent's beliefs and ballot immediately afterwards.
 *
 * UI strings come from i18n.js; game data is never translated.
 */

const api = {
  async get(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error((await r.text()).slice(0, 300));
    return r.json();
  },
  async post(path, body) {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw new Error((await r.text()).slice(0, 300));
    return r.json();
  },
  async del(path) {
    const r = await fetch(path, { method: "DELETE" });
    return r.json();
  },
};

const el = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ------------------------------------------------------------ navigation */

document.querySelectorAll("nav button").forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll("nav button").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    el(btn.dataset.page).classList.add("active");
    if (btn.dataset.page === "config") loadConfig();
    if (btn.dataset.page === "experiments") loadExperiments();
  };
});

document.querySelectorAll(".lang button").forEach((b) => {
  b.onclick = () => {
    if (LANG.current === b.dataset.lang) return;
    LANG.current = b.dataset.lang;
    storeLang(LANG.current);
    applyStaticStrings();
    /* Re-render whatever is on screen, keeping the selected game and seat. */
    if (state.log) {
      renderMeta(); renderTimeline(); renderRounds(); renderPhases();
      renderPlayers(); renderTrace();
    }
    if (el("config").classList.contains("active")) loadConfig();
    if (el("experiments").classList.contains("active")) pollExperiment();
  };
});

/* ================================================================ REPLAY */

const state = { log: null, round: 1, player: null };

function guardLabel(cfg) {
  return (cfg.guard_layers || []).join("+") || t().guardNone;
}

/* Benign persuasion is planted through the same machinery as an attack, but it
   is legitimate play -- painting it attack-red would invert the one distinction
   this whole instrument exists to measure. */
function isBenign(payloadId) {
  const p = (state.log.planted_payloads || []).find((x) => x.payload_id === payloadId);
  return Boolean(p && p.benign);
}

async function loadGames() {
  const games = await api.get("/api/games?limit=100");
  const select = el("game-select");
  select.innerHTML = games.map((g) => {
    const guard = guardLabel(g.config) + (g.config.evidence_forced ? "+E" : "");
    const mode = g.config.benign_persuasion ? "benign" : g.config.attack_enabled ? "attack" : "clean";
    return `<option value="${g.game_id}">${g.game_id} · ${t().metaSeed} ${g.seed} · ${
      guard} · ${mode} · ${esc(t().winner[g.outcome.winner] || g.outcome.winner || "?")}</option>`;
  }).join("");
  if (games.length) loadGame(games[0].game_id);
  else el("game-meta").textContent = t().noGames;
}

async function loadGame(id) {
  state.log = await api.get(`/api/games/${id}`);
  state.round = state.log.rounds.length ? state.log.rounds[0].round : 1;
  state.player = null;
  renderMeta(); renderTimeline(); renderRounds(); renderPhases(); renderPlayers(); renderTrace();
}

function renderMeta() {
  const L = t(), l = state.log, o = l.outcome;
  // A scripted run must never be mistaken for model behaviour. The model name
  // alone is easy to skim past, so it gets said outright.
  const seats = Object.values((l.config || {}).seat_models || {});
  const scripted = seats.length ? seats.every((m) => m === "mock") : l.config.model === "mock";
  el("game-meta").innerHTML =
    (scripted ? `<div class="mockbar"><b>${esc(L.mockBadge)}</b> ${esc(L.mockNote)}</div>` : "") +
    `${L.metaGame} <b>${esc(l.game_id)}</b> &nbsp; ${L.metaSeed}=${l.seed} &nbsp; ` +
    `${L.metaModel}=${esc(l.config.model)} &nbsp; ` +
    `${L.metaGuard}=${esc(guardLabel(l.config))}${l.config.evidence_forced ? "+E" : ""} &nbsp; ` +
    `${L.metaSteps}${l.config.max_react_steps} &nbsp; ` +
    `${L.metaWinner}=<b>${esc(L.winner[o.winner] || o.winner)}</b> &nbsp; ` +
    `${L.metaTokens}=${o.total_prompt_tokens + o.total_completion_tokens} &nbsp; ` +
    `${L.metaCost}=$${(o.total_cost_usd || 0).toFixed(4)} &nbsp; ${o.total_duration_s}s` +
    (o.crashed ? ` <span class="attack">${esc(L.crashed(o.crash_reason))}</span>` : "");
}

/* One tick per planted payload; filled when some agent's ballot ended up on
   the payload's target that round. Scanning the row tells you what happened in
   a game before reading a single trace. */
function renderTimeline() {
  const l = state.log, planted = l.planted_payloads || [], parts = [];
  l.rounds.forEach((rnd, i) => {
    if (i) parts.push('<span class="tick round-sep"></span>');
    planted.filter((p) => p.round === rnd.round).forEach((p) => {
      const hit = rnd.agents.some(
        (a) => a.task === "vote" && a.vote === p.target &&
               (a.read_payloads || []).some((r) => r.payload_id === p.payload_id));
      parts.push(`<span class="tick ${hit ? "hit" : ""} ${p.benign ? "benign" : ""}" title="${
        t().roundTab(rnd.round)}: ${esc(p.payload_id)} ${p.attacker}→${p.target}"></span>`);
    });
  });
  el("timeline").innerHTML = parts.join("");
}

function renderRounds() {
  el("rounds").innerHTML = state.log.rounds.map((r) =>
    `<button data-round="${r.round}" class="${r.round === state.round ? "active" : ""}">${
      esc(t().roundTab(r.round))}</button>`).join("");
  el("rounds").querySelectorAll("button").forEach((b) => {
    b.onclick = () => {
      state.round = Number(b.dataset.round);
      renderRounds(); renderPhases(); renderPlayers(); renderTrace();
    };
  });
}

/* The round as a game record: everything that happened, in the order it
 * happened, with nobody's turn hidden behind a click on their seat.
 *
 * Night first -- each wolf's pick and what the pack settled on, then the seer,
 * then the witch -- then dawn, then speeches in seat order, then the ballot.
 * Each line opens into that turn's reasoning, so the record reads end to end at
 * a glance and still goes as deep as the trace does. */
function renderPhases() {
  const L = t(), rnd = currentRound();
  const injected = new Set((rnd.injected_payloads || []).map((x) => x.attacker));

  const nightRows = (rnd.night_turns || []).map((turn) => {
    const na = turn.night_action || {};
    const act = esc(L.nightAction[na.action] || na.action || "-");
    return turnRecord(turn, esc(L.nightLine(
      turn.player_id, seatRole(turn.player_id), act, na.target || 0, na.outcome || "")));
  }).join("") || `<div class="pline muted">-</div>`;

  const deaths = (rnd.night_deaths || []).length
    ? L.diedTonight(rnd.night_deaths.map((p) => esc(L.seatId(p))).join(", "))
    : L.nobodyDied;

  const speechRows = rnd.agents
    .filter((a) => a.task === "speak")
    .sort((a, b) => a.speech_order - b.speech_order)
    .map((a) => {
      const flag = injected.has(a.player_id)
        ? `<span class="attack"> · ${esc(L.carriedPayload)}</span>` : "";
      const head = `<b>#${a.speech_order + 1} ${esc(L.seatId(a.player_id))}</b> ` +
                   `<span class="muted">${seatRole(a.player_id)}</span>${flag}`;
      const said = markSeatRefs(highlightPayloads(a.speech || ""), a.player_id);
      return turnRecord(a, `${head}<div class="said">${said}</div>`);
    }).join("") || `<div class="pline muted">-</div>`;

  const voteRows = rnd.agents.filter((a) => a.task === "vote").map((a) => {
    const blocked = (a.guard_blocks || []).length
      ? ` <span class="guard">${esc(L.blockedTimes(a.guard_blocks.length))}</span>` : "";
    return turnRecord(a, esc(L.voteLine(a.player_id, a.vote ?? null)) + blocked);
  }).join("");

  el("phases").innerHTML = `
    <div class="phase">
      <div class="phase-label">${esc(L.phaseNight)}</div>
      <div class="phase-body">${nightRows}
        <div class="pline dawn">${esc(L.dawn)}: ${esc(deaths)}</div></div>
    </div>
    ${rnd.agents.length ? `
    <div class="phase">
      <div class="phase-label">${esc(L.phaseSpeech)}</div>
      <div class="phase-body">${speechRows}</div>
    </div>
    <div class="phase">
      <div class="phase-label">${esc(L.phaseVote)}</div>
      <div class="phase-body">${voteRows}
        <div class="pline dawn">${esc(L.exiledLine(rnd.exiled))}</div></div>
    </div>` : `<div class="phase"><div class="phase-label"></div>
      <div class="phase-body"><div class="pline dawn">${esc(L.gameEndsHere)}</div></div></div>`}`;
}

function seatRole(pid) {
  const L = t(), roles = state.log.ground_truth.roles || {};
  return esc(L.role[roles[pid]] || roles[pid] || "?");
}

/* Mark seat references inside a speech.
 *
 * A record whose entire content is "who said what about whom" has to make the
 * two kinds of number distinguishable: the seat that is speaking, and the seats
 * it is speaking about. Read quickly, "2 号 女巫: ...where I stand on player 6"
 * otherwise scans as a claim to be player 6.
 *
 * Applied to the rendered HTML, but only to the parts between tags -- the
 * payload highlighter has already inserted <mark> elements with title
 * attributes, and rewriting inside those would corrupt them.
 */
function markSeatRefs(html, self) {
  return html.split(/(<[^>]*>)/).map((part) => {
    if (part.startsWith("<")) return part;
    const wrap = (text, n) =>
      `<span class="ref${Number(n) === self ? " self" : ""}">${text}</span>`;
    return part
      // "player 6" / "p6" / "6 号" -- the whole phrase reads as one reference
      .replace(/\b(?:player\s*|p)(\d)\b|(\d)\s*号/gi, (m, a, b) => {
        const n = Number(a || b);
        return n >= 1 && n <= 8 ? wrap(m, n) : m;
      })
      // "vote 6" -- wrap the number only, so the verb stays plain
      .replace(/\b(vote\s+(?:for\s+)?)(\d)\b/gi, (m, lead, d) =>
        Number(d) >= 1 && Number(d) <= 8 ? lead + wrap(d, d) : m);
  }).join("");
}

/* One line of the record, opening into that turn's reasoning. */
function turnRecord(turn, summary) {
  const L = t();
  const steps = (turn.react_trace || []).map((s) => {
    const cls = s.injected ? "injected" : s.guard_blocked ? "blocked" : "";
    const flag = s.injected
      ? `<span class="pill attack">${esc(L.injectedPill)}</span>`
      : s.guard_blocked
      ? `<span class="pill guard">${esc(L.blockPill(L.block[s.block_reason] || s.block_reason))}</span>`
      : "";
    return `<div class="step ${cls}">
      <div class="head">${s.step} &rarr; <b>${esc(s.action)}</b>
        <span class="muted">${esc(JSON.stringify(s.args || {}))}</span> ${flag}</div>
      ${s.thought ? `<div class="obs">${esc(L.thought)}: ${esc(s.thought)}</div>` : ""}
      ${s.observation ? `<div class="obs">${highlightPayloads(s.observation)}</div>` : ""}
    </div>`;
  }).join("");
  const diff = beliefDiff(turn.belief_before, turn.belief_after);
  const exposure = (turn.read_payloads || []).map((r) =>
    `<span class="pill ${isBenign(r.payload_id) ? "guard" : "attack"}">${
      esc(L.exposurePill(r.payload_id, L.channel[r.channel] || r.channel))}</span>`).join(" ");

  return `<details class="record">
    <summary class="pline">${summary}</summary>
    <div class="record-body">
      <div class="muted mono">${esc(L.modelLabel)} ${esc(turn.model || "?")} &nbsp;
        ${esc(L.steps(turn.steps_used))} &nbsp; ${esc(L.tokens(turn.total_tokens))}
        ${turn.retries ? `&nbsp; ${esc(L.retries(turn.retries))}` : ""}
        ${turn.forced_terminal ? `&nbsp; <span class="guard">${esc(L.forced)}</span>` : ""}</div>
      ${exposure ? `<div style="margin:6px 0">${esc(L.exposedTo)}${exposure}</div>` : ""}
      ${steps}
      ${diff ? `<div class="beliefdiff" style="margin-top:6px">${esc(L.deltaLead)}: ${diff}</div>` : ""}
    </div>
  </details>`;
}

function currentRound() {
  return state.log.rounds.find((r) => r.round === state.round) || { agents: [], alive: [] };
}

function renderPlayers() {
  const L = t(), rnd = currentRound(), roles = state.log.ground_truth.roles || {};
  const seatModels = (state.log.config || {}).seat_models || {};
  const attackers = new Set((state.log.planted_payloads || [])
    .filter((p) => p.round === rnd.round).map((p) => p.attacker));
  const items = [];
  for (let pid = 1; pid <= 8; pid++) {
    const turns = rnd.agents.filter((a) => a.player_id === pid);
    const alive = (rnd.alive || []).includes(pid);
    const exposed = turns.some((x) =>
      (x.read_payloads || []).some((r) => !isBenign(r.payload_id)));
    const persuaded = turns.some((x) =>
      (x.read_payloads || []).some((r) => isBenign(r.payload_id)));
    const blocked = turns.some((x) => (x.guard_blocks || []).length);
    const human = turns.some((x) => x.is_human);
    items.push(
      `<li data-pid="${pid}" class="${state.player === pid ? "selected" : ""} ${alive ? "" : "dead"}">
        <span class="flag attack">${attackers.has(pid) ? "&#9679;" : ""}</span>
        <span>${esc(L.seatId(pid))}</span>
        <span class="muted">${esc(L.role[roles[pid]] || roles[pid] || "?")}${human ? L.human : ""}</span>
        <span class="muted seat-model">${esc(human ? "" : (seatModels[pid] || ""))}</span>
        <span style="margin-left:auto">
          ${exposed ? '<span class="attack">&#9888;</span>' : ""}
          ${persuaded ? '<span class="guard">&#9733;</span>' : ""}
          ${blocked ? '<span class="guard">&#9635;</span>' : ""}
        </span>
      </li>`);
  }
  el("players").innerHTML = items.join("");
  el("players").querySelectorAll("li").forEach((li) => {
    li.onclick = () => { state.player = Number(li.dataset.pid); renderPlayers(); renderTrace(); };
  });

  const counts = Object.entries(rnd.vote_counts || {})
    .map(([tgt, c]) => L.votes(tgt, c)).join("  ") || L.noVotes;
  el("round-summary").innerHTML =
    `${L.hNight}: ${(rnd.night_turns || []).length} ${L.nightTurn}<br>` +
    `${L.sumDeaths}: ${(rnd.night_deaths || []).length
      ? rnd.night_deaths.map((p) => esc(L.seatId(p))).join(", ") : L.none}<br>` +
    `${L.sumTally}: ${esc(counts)}<br>` +
    `${L.sumExiled}: ${rnd.exiled ? esc(L.seatId(rnd.exiled)) : L.nobody}`;
}

/* Highlight the exact payload text inside whatever the agent read. */
function highlightPayloads(text) {
  let out = esc(text);
  (state.log.planted_payloads || []).forEach((p) => {
    const words = String(p.text).split(/\s+/);
    const frag = words.slice(Math.floor(words.length / 3), Math.floor(words.length / 3) + 5).join(" ");
    if (frag && out.includes(esc(frag))) {
      out = out.replace(esc(frag), `<mark title="payload ${esc(p.payload_id)}">${esc(frag)}</mark>`);
    }
  });
  return out;
}

function renderTrace() {
  const L = t(), rnd = currentRound();
  if (!state.player) {
    el("trace-title").textContent = L.hTrace;
    el("trace").innerHTML = `<div class="empty">${esc(L.tracePick)}</div>`;
    return;
  }
  const turns = [
    ...(rnd.night_turns || []).filter((a) => a.player_id === state.player),
    ...rnd.agents.filter((a) => a.player_id === state.player),
  ];
  el("trace-title").textContent = L.traceHeadFor(state.player, rnd.round);
  if (!turns.length) {
    el("trace").innerHTML = `<div class="empty">${esc(L.traceNone)}</div>`;
    return;
  }

  el("trace").innerHTML = turns.map((turn) => {
    const steps = (turn.react_trace || []).map((s) => {
      const cls = s.injected ? "injected" : s.guard_blocked ? "blocked" : "";
      const flag = s.injected
        ? `<span class="pill attack">${esc(L.injectedPill)}</span>`
        : s.guard_blocked
        ? `<span class="pill guard">${esc(L.blockPill(L.block[s.block_reason] || s.block_reason))}</span>`
        : "";
      return `<div class="step ${cls}">
          <div class="head">${s.step} &rarr; <b>${esc(s.action)}</b>
            <span class="muted">${esc(JSON.stringify(s.args || {}))}</span> ${flag}</div>
          ${s.thought ? `<div class="obs">${esc(L.thought)}: ${esc(s.thought)}</div>` : ""}
          ${s.observation ? `<div class="obs">${highlightPayloads(s.observation)}</div>` : ""}
        </div>`;
    }).join("");

    const diff = beliefDiff(turn.belief_before, turn.belief_after);
    const exposure = (turn.read_payloads || []).map((r) =>
      `<span class="pill ${isBenign(r.payload_id) ? "guard" : "attack"}">${
        esc(L.exposurePill(r.payload_id, L.channel[r.channel] || r.channel))}</span>`
    ).join(" ");

    return `<div style="margin-bottom:18px">
      <div class="muted mono">${esc(turn.task === "night" ? L.nightTurn : (L.task[turn.task] || turn.task))} &nbsp;
        <span class="pill">${esc(L.modelLabel)} ${esc(turn.model || "?")}</span> &nbsp;
        ${esc(L.steps(turn.steps_used))} &nbsp; ${esc(L.tokens(turn.total_tokens))}
        ${turn.retries ? `&nbsp; ${esc(L.retries(turn.retries))}` : ""}
        ${turn.forced_terminal ? `&nbsp; <span class="guard">${esc(L.forced)}</span>` : ""}
        ${turn.fallback_used ? `&nbsp; <span class="attack">${esc(L.fallbackUsed(turn.fallback_used))}</span>` : ""}</div>
      ${exposure ? `<div style="margin:6px 0">${esc(L.exposedTo)}${exposure}</div>` : ""}
      ${steps}
      ${diff ? `<div class="beliefdiff" style="margin-top:8px">${esc(L.deltaLead)}: ${diff}</div>` : ""}
      ${turn.speech ? `<div class="speech">${highlightPayloads(turn.speech)}</div>` : ""}
      ${turn.task === "vote"
        ? `<div class="mono">${esc(L.ballot)}<b>${turn.vote ? esc(L.seatId(turn.vote)) : esc(L.abstained)}</b></div>`
        : ""}
      ${turn.night_action
        ? `<div class="mono">${esc(L.nightAction[turn.night_action.action] || turn.night_action.action)}
           <b>${esc(L.nightOn(turn.night_action.target))}</b>
           <span class="muted">${esc(turn.night_action.outcome || "")}</span></div>`
        : ""}
    </div>`;
  }).join("");
}

function beliefDiff(before, after) {
  const L = t(), parts = [];
  Object.keys(after || {}).forEach((pid) => {
    const b = before && before[pid] ? before[pid].suspicion : null;
    const a = after[pid].suspicion;
    if (b !== null && Math.abs(a - b) > 1e-9) {
      parts.push(`<span class="up">${esc(L.deltaRow(pid, b.toFixed(2), a.toFixed(2)))}</span>
        <span class="muted">(${esc(after[pid].reason || "")})</span>`);
    }
  });
  return parts.join("<br>");
}

el("game-select").onchange = (e) => loadGame(e.target.value);

el("run-game").onclick = async () => {
  const guard = el("new-guard").value ? el("new-guard").value.split(",") : [];
  const started = await api.post("/api/games", {
    seed: Number(el("new-seed").value),
    guard_layers: guard,
    attack_enabled: true,
  });
  el("game-meta").textContent = t().running(started.config);
  const source = new EventSource(`/api/games/${started.stream_id}/stream`);
  source.addEventListener("round_start", (e) => {
    el("game-meta").textContent = t().runningRound(JSON.parse(e.data).round);
  });
  source.addEventListener("done", async (e) => {
    source.close();
    await loadGames();
    loadGame(JSON.parse(e.data).game_id);
  });
  source.addEventListener("end", () => source.close());
  source.addEventListener("error", () => source.close());
};

/* ================================================================ CONFIG */

async function loadConfig() {
  const L = t();
  const [providers, models] = await Promise.all([
    api.get("/api/providers"),
    api.get("/api/models"),
  ]);

  el("providers").innerHTML =
    `<tr><th>${L.colName}</th><th>${L.colUrl}</th><th>${L.colKey}</th><th></th></tr>` +
    (providers.map((p) => `<tr>
        <td>${esc(p.name)} <span class="muted">· ${({claude_cli: L.kindCli, anthropic: L.kindAnthropic})[p.kind] || L.kindRelay}</span></td><td class="mono">${esc(p.base_url)}</td>
        <td class="mono muted">${esc(p.api_key_masked)}</td>
        <td><button class="action danger" data-del="${p.id}">${L.btnDelete}</button></td>
      </tr>`).join("") || `<tr><td colspan="4" class="muted">${L.empty}</td></tr>`);

  el("providers").querySelectorAll("[data-del]").forEach((b) => {
    b.onclick = async () => { await api.del(`/api/providers/${b.dataset.del}`); loadConfig(); };
  });

  el("m-provider").innerHTML = providers.map((p) =>
    `<option value="${p.id}">${esc(p.name)}</option>`).join("");
  el("x-model").innerHTML = `<option value="">${L.mockModel}</option>` +
    models.map((m) => `<option value="${m.id}">${esc(m.display_name)}</option>`).join("");

  el("models").innerHTML =
    `<tr><th>${L.colDisplay}</th><th>${L.colModel}</th><th>${L.colGroup}</th>` +
    `<th>${L.colTools}</th><th>${L.colMode}</th><th>${L.colProbe}</th><th></th></tr>` +
    (models.map((m) => `<tr>
        <td>${esc(m.display_name)}</td>
        <td class="mono">${esc(m.model_name)}</td>
        <td class="mono muted">${esc(m.group || "-")}</td>
        <td>${m.supports_tools === null ? `<span class="muted">${L.toolsUntested}</span>`
             : m.supports_tools ? `<span class="guard">${L.toolsNative}</span>`
             : `<span class="attack">${L.toolsNone}</span>`}</td>
        <td class="mono">${esc(m.tool_mode)}</td>
        <td><button class="action" data-probe="${m.id}">${L.btnTest}</button>
            <span id="probe-${m.id}" class="mono muted"></span></td>
        <td><button class="action danger" data-delm="${m.id}">${L.btnDelete}</button></td>
      </tr>`).join("") || `<tr><td colspan="7" class="muted">${L.empty}</td></tr>`);

  el("models").querySelectorAll("[data-delm]").forEach((b) => {
    b.onclick = async () => { await api.del(`/api/models/${b.dataset.delm}`); loadConfig(); };
  });
  el("models").querySelectorAll("[data-probe]").forEach((b) => {
    b.onclick = () => probeModel(b.dataset.probe);
  });
}

/* The probe result is the point of the config page: three plain answers
 * (reachable / tools / latency) and, on failure, a fix rather than the
 * gateway's own error text. */
async function probeModel(id) {
  const L = t(), out = el(`probe-${id}`);
  out.textContent = L.testing;
  try {
    const r = await api.post(`/api/models/${id}/probe`);
    const bits = [
      r.reachable ? `<span class="guard">${esc(L.reachable(r.latency_ms))}</span>`
                  : `<span class="attack">${esc(L.unreachable)}</span>`,
      r.native_tools ? `<span class="guard">${esc(L.nativeTools)}</span>`
                     : `<span class="belief">${esc(L.jsonFallback)}</span>`,
      r.reports_usage ? "" : `<span class="belief">${esc(L.noUsage)}</span>`,
    ].filter(Boolean);
    out.innerHTML = bits.join(" &middot; ");
    if (r.error) out.innerHTML += `<div class="hint">${esc(r.hint || r.error)}</div>`;
    (r.notes || []).forEach((n) => { out.innerHTML += `<div class="hint">${esc(n)}</div>`; });
    loadConfig();
  } catch (e) {
    let hint = String(e.message);
    try { hint = JSON.parse(hint).detail.hint || hint; } catch (_) { /* raw text */ }
    out.innerHTML = `<span class="attack">${esc(L.probeFailed)}</span><div class="hint">${esc(hint)}</div>`;
  }
}

el("add-provider").onclick = async () => {
  try {
    const kind = el("p-kind").value;
    const created = await api.post("/api/providers", {
      name: el("p-name").value || kind,
      base_url: el("p-url").value || (kind === "anthropic" ? "https://api.anthropic.com" : ""),
      api_key: el("p-key").value,
      kind,
    });
    el("p-key").value = "";
    el("provider-msg").textContent = created.warning || t().providerSaved;
    loadConfig();
  } catch (e) {
    el("provider-msg").innerHTML = `<span class="attack">${esc(e.message)}</span>`;
  }
};

el("add-model").onclick = async () => {
  try {
    await api.post("/api/models", {
      provider_id: el("m-provider").value,
      display_name: el("m-display").value || el("m-name").value,
      model_name: el("m-name").value,
      group: el("m-group").value || null,
    });
    el("model-msg").textContent = t().modelSaved;
    loadConfig();
  } catch (e) {
    el("model-msg").innerHTML = `<span class="attack">${esc(e.message)}</span>`;
  }
};

/* =========================================================== EXPERIMENTS */

let currentExperiment = null;
let pollTimer = null;

async function loadExperiments() {
  const list = await api.get("/api/experiments");
  if (list.length && !currentExperiment) currentExperiment = list[0].id;
  if (currentExperiment) pollExperiment();
}

el("start-exp").onclick = async () => {
  const exp = await api.post("/api/experiments", {
    name: el("x-name").value,
    seeds: Number(el("x-seeds").value),
    workers: Number(el("x-workers").value),
    model_id: el("x-model").value || null,
    include_benign: el("x-benign").checked,
  });
  currentExperiment = exp.id;
  pollExperiment();
};

el("stop-exp").onclick = async () => {
  if (currentExperiment) await api.post(`/api/experiments/${currentExperiment}/stop`);
};

async function pollExperiment() {
  clearTimeout(pollTimer);
  if (!currentExperiment) return;
  const L = t();
  const exp = await api.get(`/api/experiments/${currentExperiment}`);
  const p = exp.progress;
  el("exp-progress").innerHTML =
    `<b>${esc(exp.name)}</b> &nbsp; ${esc(L.status[exp.status] || exp.status)}<br>` +
    `${esc(L.expLine(p.done, p.total, p.crashed, p.tokens))}` +
    `<progress value="${p.done}" max="${p.total || 1}"></progress>`;
  if (p.done) renderExperimentResults();
  if (["running", "queued", "stopping"].includes(exp.status)) {
    pollTimer = setTimeout(pollExperiment, 1500);
  }
}

async function renderExperimentResults() {
  const L = t();
  const data = await api.get(`/api/experiments/${currentExperiment}/metrics`);
  const rows = Object.entries(data.arms).map(([label, m]) => `<tr>
      <td class="mono">${esc(label)}</td>
      <td class="num">${pct((m.injection.speech || {}).hijack_rate)}</td>
      <td class="num">${pct((m.injection.tool_return || {}).hijack_rate)}</td>
      <td class="num">${pct((m.injection.speech || {}).latent_rate)}</td>
      <td class="num">${pct(m.overdefense.false_block_rate)}</td>
      <td class="num">${pct(m.village_win_rate)}</td>
      <td class="num">${m.mean_tokens_per_game ?? "-"}</td>
      <td class="num">${m.crashed}</td>
    </tr>`).join("");
  el("exp-results").innerHTML =
    `<tr><th>${L.colGuard}</th><th class="num">${L.colHijackSpeech}</th>` +
    `<th class="num">${L.colHijackTool}</th><th class="num">${L.colLatent}</th>` +
    `<th class="num">${L.colFalseBlock}</th><th class="num">${L.colVillageWin}</th>` +
    `<th class="num">${L.colTokens}</th><th class="num">${L.colCrashed}</th></tr>` + rows;
}

const pct = (v) => (v === null || v === undefined ? "-" : (100 * v).toFixed(1) + "%");

applyStaticStrings();
loadGames();
