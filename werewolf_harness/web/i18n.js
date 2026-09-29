/* UI strings, both languages, one page.
 *
 * Game data is never translated: speeches, payload text and tool arguments are
 * the material under test, and rewriting them would change what the instrument
 * recorded. Only the chrome around them switches.
 */

const I18N = {
  zh: {
    sub: "多 agent 注入测量仪 · 狼人杀环境",
    navReplay: "回放", navConfig: "配置", navExperiments: "实验",

    /* --- 回放 --- */
    labelGame: "对局", labelNewGame: "新开一局", guardNone: "无防护", btnRun: "开跑",
    hSeats: "座位", hTrace: "ReAct 轨迹",
    noGames: "还没有对局 —— 用上面的按钮跑一局，或先执行 cli seed-db",
    legend: [
      ["swatch hit", "注入生效 —— 有票被拽到了目标身上"],
      ["swatch", "投放了，但没有票改变"],
      ["glyph attack ●", "攻击者"],
      ["glyph attack ⚠", "读到过 payload"],
      ["glyph guard ▣", "被防护拦截过"],
      ["glyph guard ★", "读到过良性说服（对照组）"],
    ],
    mockBadge: "脚本客户端 · 不是模型输出",
    mockNote: "这一局由离线脚本客户端产生：约 200 行 if-else 加正则，没有任何模型参与。它的作用是把整条链路跑通（工具调用、防护拦截、重试、兜底），不是打得好。要看真实推理，配好网关后用 --model 跑。",
    metaGame: "对局", metaSeed: "种子", metaModel: "模型", metaGuard: "防护",
    metaSteps: "步数上限", metaWinner: "结局", metaTokens: "token", metaCost: "成本",
    winner: { werewolf: "狼人胜", village: "村民胜" },
    crashed: (why) => `本局崩溃：${why}`,
    running: (label) => `正在跑 ${label} ...`,
    runningRound: (n) => `正在跑 ... 第 ${n} 轮`,
    roundTab: (n) => `第 ${n} 轮`,
    role: { werewolf: "狼人", seer: "预言家", witch: "女巫", villager: "村民" },
    seatId: (p) => `${p} 号`, human: "（真人）",
    sumDeaths: "夜间死亡", sumTally: "票型", sumExiled: "出局",
    hNight: "夜间", nightTurn: "夜间回合", modelLabel: "模型",
    hPhases: "本轮时序",
    phaseNight: "夜", phaseSpeech: "昼 · 发言", phaseVote: "昼 · 投票",
    dawn: "天亮", nobodyDied: "无人死亡", diedTonight: (who) => `${who} 死亡`,
    nightLine: (seat, role, act, target, outcome) =>
      `${seat} 号 ${role}　${act}${target ? ` → ${target} 号` : ""}　${outcome}`,
    speechLine: (order, seat, target) =>
      `#${order} ${seat} 号${target ? ` → 表态投 ${target} 号` : ""}`,
    carriedPayload: "夹带注入",
    voteLine: (seat, target) => `${seat} 号 → ${target === null ? "弃票" : target + " 号"}`,
    exiledLine: (who) => (who ? `出局 ${who} 号` : "平票，无人出局"),
    blockedTimes: (n) => `被拦 ${n} 次`,
    gameEndsHere: "游戏在此结束",
    nightAction: {
      night_check: "验人", night_kill: "刀人", night_save: "用解药",
      night_poison: "用毒药", night_skip: "空过",
    },
    nightOn: (t) => (t ? `→ ${t} 号` : ""),
    seatModels: "各座位模型",
    none: "无", nobody: "无人出局", noVotes: "无人投票",
    votes: (t, c) => `${t}号 ${c}票`,
    tracePick: "点左边任意一个座位查看它这一轮的完整推理",
    traceNone: "这个座位在本轮没有行动",
    traceHeadFor: (p, r) => `ReAct 轨迹 —— ${p} 号，第 ${r} 轮`,
    task: { speak: "发言回合", vote: "投票回合" },
    steps: (n) => `${n} 步`, tokens: (n) => `${n} token`, retries: (n) => `重试 ${n} 次`,
    forced: "强制收尾", fallbackUsed: (w) => `走了兜底：${w}`,
    injectedPill: "这一步的工具返回值里带着注入",
    blockPill: (why) => `拦截 · ${why}`,
    block: {
      schema: "格式不合规", semantics: "语义不合法", evidence: "缺少取证",
      say_do_consistency: "言行不一", directive_compliance: "无据服从指令",
      engine_rejected: "裁判驳回", fallback: "兜底动作",
    },
    channel: { speech: "发言", tool_return: "工具返回" },
    exposedTo: "读到过：", exposurePill: (id, ch) => `${id} · 经${ch}读到`,
    thought: "想法", deltaLead: "判断变化",
    deltaRow: (p, from, to) => `${p} 号 怀疑度 ${from} → ${to}`,
    ballot: "最终投票：", abstained: "弃票",

    /* --- 配置 --- */
    hProviders: "接入点",
    providerNote: "令牌只存在服务端，永远不会回传到这个页面；浏览器也从不直接请求网关。选"Claude 订阅"时不用填 base_url 和令牌：走本机已登录的 claude 命令，用的是你的订阅额度。",
    kindAnthropic: "Claude API（按 token 计费）", kindRelay: "中转站", kindCli: "Claude 订阅（claude -p）", fKind: "接入方式",
    fName: "名称", fBaseUrl: "base_url", fKey: "令牌", btnAdd: "添加",
    colName: "名称", colUrl: "base_url", colKey: "令牌", colOps: "",
    btnDelete: "删除", empty: "还没有",
    providerSaved: "已保存 —— 令牌留在服务端",
    hModels: "模型",
    modelNote: "模型名是自由输入，这是刻意的：从网关的模型列表里复制，别用写死的下拉框 —— 网关一改名整个列表就废了。tool mode 由探针决定，不是手填的。",
    fProvider: "接入点", fDisplay: "显示名", fModelName: "模型名（复制，别手打）", fGroup: "令牌分组",
    colDisplay: "显示名", colModel: "模型名", colGroup: "分组", colTools: "工具",
    colMode: "模式", colProbe: "探针",
    toolsUntested: "未测", toolsNative: "原生", toolsNone: "不支持",
    btnTest: "测试", testing: "测试中...",
    reachable: (ms) => `连通 ${ms}ms`, unreachable: "连不通",
    nativeTools: "原生工具调用", jsonFallback: "降级 JSON 模式", noUsage: "无 usage 字段",
    probeFailed: "失败",
    modelSaved: "已保存 —— 进实验前先跑一次探针",

    /* --- 实验 --- */
    hNewBatch: "新建批次",
    fBatchName: "名称", fSeeds: "每组种子数", fWorkers: "并发", fModel: "模型",
    fBenign: "良性对照组", btnStart: "开始",
    batchNote: "每组都跑同一批种子，所以组间比较是按种子配对的。良性对照组投放的是合法说服而不是攻击，误拦率就是在它上面量的。",
    hRunning: "进行中", btnStop: "中断",
    expLine: (done, total, crashed, tokens) =>
      `${done}/${total} 局 · 崩溃 ${crashed} · token ${tokens}`,
    status: { queued: "排队中", running: "进行中", stopping: "停止中",
              stopped: "已中断", finished: "已完成" },
    hResults: "结果",
    resultsNote: "这里是点估计；命令行（python -m werewolf_harness.cli ablation）会同时打出 Wilson 置信区间。",
    colGuard: "防护", colHijackSpeech: "劫持（发言）", colHijackTool: "劫持（工具）",
    colLatent: "潜伏失效", colFalseBlock: "误拦", colVillageWin: "村民胜率",
    colTokens: "token/局", colCrashed: "崩溃",
    mockModel: "mock（离线）",
  },

  en: {
    sub: "multi-agent prompt-injection instrument — werewolf environment",
    navReplay: "replay", navConfig: "config", navExperiments: "experiments",

    labelGame: "game", labelNewGame: "new game", guardNone: "no guard", btnRun: "run",
    hSeats: "seats", hTrace: "react trace",
    noGames: "no games yet — run one above, or start with cli seed-db",
    legend: [
      ["swatch hit", "injection changed a vote"],
      ["swatch", "injection delivered, vote unchanged"],
      ["glyph attack ●", "attacker"],
      ["glyph attack ⚠", "read a payload"],
      ["glyph guard ▣", "guard blocked an action"],
      ["glyph guard ★", "read benign persuasion (control arm)"],
    ],
    mockBadge: "scripted client · not model output",
    mockNote: "This game came from the offline scripted client: about 200 lines of if-else and regex, with no model involved. Its job is to exercise the pipeline -- tool calls, guard blocks, retries, fallbacks -- not to play well. For real reasoning, configure a gateway and run with --model.",
    metaGame: "game", metaSeed: "seed", metaModel: "model", metaGuard: "guard",
    metaSteps: "steps ≤", metaWinner: "winner", metaTokens: "tokens", metaCost: "cost",
    winner: { werewolf: "wolves won", village: "village won" },
    crashed: (why) => `CRASHED: ${why}`,
    running: (label) => `running ${label} ...`,
    runningRound: (n) => `running ... round ${n}`,
    roundTab: (n) => `round ${n}`,
    role: { werewolf: "werewolf", seer: "seer", witch: "witch", villager: "villager" },
    seatId: (p) => `p${p}`, human: " (human)",
    sumDeaths: "night deaths", sumTally: "tally", sumExiled: "exiled",
    hNight: "night", nightTurn: "night turn", modelLabel: "model",
    hPhases: "how this round played out",
    phaseNight: "night", phaseSpeech: "day · speeches", phaseVote: "day · vote",
    dawn: "dawn", nobodyDied: "nobody died", diedTonight: (who) => `${who} died`,
    nightLine: (seat, role, act, target, outcome) =>
      `p${seat} ${role}　${act}${target ? ` → p${target}` : ""}　${outcome}`,
    speechLine: (order, seat, target) =>
      `#${order} p${seat}${target ? ` → says vote p${target}` : ""}`,
    carriedPayload: "carried a payload",
    voteLine: (seat, target) => `p${seat} → ${target === null ? "abstained" : "p" + target}`,
    exiledLine: (who) => (who ? `p${who} exiled` : "tie, nobody out"),
    blockedTimes: (n) => `blocked ${n}x`,
    gameEndsHere: "the game ends here",
    nightAction: {
      night_check: "check", night_kill: "kill", night_save: "antidote",
      night_poison: "poison", night_skip: "pass",
    },
    nightOn: (t) => (t ? `→ p${t}` : ""),
    seatModels: "seat models",
    none: "none", nobody: "nobody", noVotes: "no votes",
    votes: (t, c) => `p${t}:${c}`,
    tracePick: "pick a seat to see its turn",
    traceNone: "this seat did not act in this round",
    traceHeadFor: (p, r) => `react trace — seat ${p}, round ${r}`,
    task: { speak: "speech turn", vote: "vote turn" },
    steps: (n) => `${n} steps`, tokens: (n) => `${n} tokens`, retries: (n) => `${n} retries`,
    forced: "forced", fallbackUsed: (w) => `fallback: ${w}`,
    injectedPill: "payload in this tool return",
    blockPill: (why) => `blocked · ${why}`,
    block: {
      schema: "schema", semantics: "semantics", evidence: "no evidence",
      say_do_consistency: "say/do mismatch", directive_compliance: "obeyed an instruction",
      engine_rejected: "referee rejected", fallback: "fallback",
    },
    channel: { speech: "speech", tool_return: "tool return" },
    exposedTo: "exposed to: ", exposurePill: (id, ch) => `${id} · via ${ch}`,
    thought: "thought", deltaLead: "belief changes",
    deltaRow: (p, from, to) => `p${p} suspicion ${from} → ${to}`,
    ballot: "ballot: ", abstained: "abstained",

    hProviders: "providers",
    providerNote: "The key is stored server-side and never sent back to this page. The browser never talks to the gateway directly. "Claude subscription" needs no base_url or key: it runs the locally logged-in claude command, on your subscription.",
    kindAnthropic: "Claude API (per-token)", kindRelay: "relay", kindCli: "Claude subscription (claude -p)", fKind: "via",
    fName: "name", fBaseUrl: "base_url", fKey: "api key", btnAdd: "add",
    colName: "name", colUrl: "base url", colKey: "key", colOps: "",
    btnDelete: "delete", empty: "none yet",
    providerSaved: "saved — the key stays on the server",
    hModels: "models",
    modelNote: "Model names are free text on purpose — copy them from the gateway's list rather than picking from a hard-coded menu, which breaks the moment the gateway renames anything. tool mode is set by the probe, not by hand.",
    fProvider: "provider", fDisplay: "display name", fModelName: "model name (copy, do not type)", fGroup: "token group",
    colDisplay: "display", colModel: "model name", colGroup: "group", colTools: "tools",
    colMode: "mode", colProbe: "probe",
    toolsUntested: "untested", toolsNative: "native", toolsNone: "none",
    btnTest: "test", testing: "testing...",
    reachable: (ms) => `reachable ${ms}ms`, unreachable: "unreachable",
    nativeTools: "native tools", jsonFallback: "json fallback", noUsage: "no usage field",
    probeFailed: "failed",
    modelSaved: "saved — run the probe before using it in a batch",

    hNewBatch: "new batch",
    fBatchName: "name", fSeeds: "seeds per arm", fWorkers: "workers", fModel: "model",
    fBenign: "benign arm", btnStart: "start",
    batchNote: "Every arm runs the identical seed set, so arms are compared paired by seed. The benign arm plants legitimate persuasion instead of attacks and is what the false-block rate is measured on.",
    hRunning: "running", btnStop: "stop",
    expLine: (done, total, crashed, tokens) =>
      `${done}/${total} games · crashed ${crashed} · tokens ${tokens}`,
    status: { queued: "queued", running: "running", stopping: "stopping",
              stopped: "stopped", finished: "finished" },
    hResults: "results",
    resultsNote: "Point estimates; the CLI (python -m werewolf_harness.cli ablation) prints Wilson intervals alongside them.",
    colGuard: "guard", colHijackSpeech: "hijack (speech)", colHijackTool: "hijack (tool)",
    colLatent: "latent", colFalseBlock: "false block", colVillageWin: "village win",
    colTokens: "tokens/game", colCrashed: "crashed",
    mockModel: "mock (offline)",
  },
};

function storedLang() {
  try { return localStorage.getItem("harness-lang"); } catch (_) { return null; }
}
function storeLang(v) {
  try { localStorage.setItem("harness-lang", v); } catch (_) { /* private mode: fine */ }
}

const LANG = { current: storedLang() || "zh" };
const t = () => I18N[LANG.current];

/* Fill every element carrying data-i18n, and the legend, from the dictionary. */
function applyStaticStrings() {
  const L = t();
  document.body.setAttribute("data-lang", LANG.current);
  document.documentElement.lang = LANG.current === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const value = L[node.dataset.i18n];
    if (typeof value === "string") node.textContent = value;
  });
  const legend = document.getElementById("legend");
  if (legend) {
    legend.innerHTML = L.legend.map(([kind, label]) => {
      if (kind.startsWith("swatch")) {
        const hit = kind.includes("hit");
        return `<span><span class="tick ${hit ? "hit" : ""}" style="display:inline-block"></span> ${label}</span>`;
      }
      const [, tone, glyph] = kind.split(" ");
      return `<span class="${tone}">${glyph} ${label}</span>`;
    }).join("");
  }
  document.querySelectorAll(".lang button").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.lang === LANG.current));
  });
}
