// Bayn dashboard client: one WebSocket, a canvas EEG plotter, simulator controls.
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  channels: ["TP9", "AF7", "AF8", "TP10"],
  fs: 256,
  seconds: 5,
  scale: 50,          // µV at the edge of each lane
  paused: false,
  buf: [],            // per-channel Float32Array ring buffers
  write: 0,
  lastT: null,        // timestamp just after the newest sample
  markers: [],        // {t, label}
  statusMsg: "",      // connection progress from the bridge, e.g. "searching for Muse-3EE1..."
};

// ------------------------------------------------------------------ helpers
function toast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove("show"), 2200);
}
const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const fmtTime = (t) => new Date(t * 1000).toLocaleTimeString([], { hour12: false });
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function resetBuffers() {
  const n = Math.round(state.fs * state.seconds);
  state.buf = state.channels.map(() => new Float32Array(n).fill(NaN));
  state.write = 0;
}

// ------------------------------------------------------------------ plotting
const canvas = $("eeg-plot");
const ctx = canvas.getContext("2d");

function resize() {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(r.width * dpr));
  canvas.height = Math.max(1, Math.round(r.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
new ResizeObserver(resize).observe(canvas);

function draw() {
  requestAnimationFrame(draw);
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (!W || !H || !state.buf.length) return;
  const nCh = state.channels.length, lane = H / nCh, n = state.buf[0].length;
  const labelW = 46, plotW = W - labelW;
  ctx.clearRect(0, 0, W, H);

  // grid: one line per second, lane separators
  ctx.strokeStyle = cssVar("--plot-grid");
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let s = 0; s <= state.seconds; s++) {
    const x = labelW + (s / state.seconds) * plotW;
    ctx.moveTo(x, 0); ctx.lineTo(x, H);
  }
  for (let c = 1; c < nCh; c++) { ctx.moveTo(labelW, c * lane); ctx.lineTo(W, c * lane); }
  ctx.stroke();

  // traces (oldest sample at left)
  const colors = [1, 2, 3, 4].map((i) => cssVar(`--trace-${i}`));
  ctx.lineWidth = 1.2;
  for (let c = 0; c < nCh; c++) {
    const mid = c * lane + lane / 2, k = (lane / 2) / state.scale, b = state.buf[c];
    ctx.strokeStyle = colors[c % colors.length];
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < n; i++) {
      const v = b[(state.write + i) % n];
      if (Number.isNaN(v)) { pen = false; continue; }
      const x = labelW + (i / (n - 1)) * plotW;
      const y = Math.max(c * lane + 1, Math.min((c + 1) * lane - 1, mid - v * k));
      pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      pen = true;
    }
    ctx.stroke();
    ctx.fillStyle = cssVar("--muted");
    ctx.font = "12px " + cssVar("--font");
    ctx.fillText(state.channels[c], 8, mid + 4);
  }

  // gesture/event markers
  if (state.lastT !== null) {
    const t0 = state.lastT - state.seconds;
    state.markers = state.markers.filter((m) => m.t > t0 - 1);
    for (const m of state.markers) {
      const x = labelW + ((m.t - t0) / state.seconds) * plotW;
      if (x < labelW || x > W) continue;
      ctx.strokeStyle = cssVar("--ok");
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = cssVar("--ok");
      ctx.fillText(m.label, x + 4, 14);
    }
  }
}

function pushEEG(msg) {
  if (state.paused) return;
  const n = state.buf[0].length, m = msg.data[0].length;
  for (let c = 0; c < state.channels.length; c++) {
    const src = msg.data[c], dst = state.buf[c];
    for (let i = 0; i < m; i++) dst[(state.write + i) % n] = src[i];
  }
  state.write = (state.write + m) % n;
  state.lastT = msg.t0 + m / msg.fs;
}

// ------------------------------------------------------------------ status UI
function setConn(kind, text) {
  const p = $("pill-conn");
  p.classList.remove("live", "down");
  if (kind) p.classList.add(kind);
  p.querySelector("span").textContent = text;
}

function renderStatus(s) {
  state.statusMsg = s.connected ? "" : s.message;
  if (s.fs && s.fs !== state.fs) {            // real device rate can differ from config
    state.fs = s.fs;
    resetBuffers();
  }
  $("pill-source").textContent = "Source: " + ({ sim: "Simulated", muse: "Muse 2", lsl: "Muse (LSL)" }[s.source] || s.source);
  $("pill-mains").textContent = "Mains: " + (s.mains_hz ? s.mains_hz + " Hz" : "measuring…");
  $("sim-card").style.display = s.supports_injection ? "" : "none";
  $("sim-state").textContent = s.supports_injection ? "active" : "–";
  const q = $("quality");
  q.innerHTML = "";
  for (const c of s.quality || []) {
    const el = document.createElement("span");
    el.className = "q " + c.level;
    el.title = c.reason || "signal looks good";
    el.innerHTML = `<i class="dot"></i>${c.channel} <small>${c.noise_uv.toFixed(1)} µV</small>`;
    q.appendChild(el);
  }
}

function logInjection(m) {
  const li = document.createElement("li");
  li.innerHTML = `<span>${m.gesture.replace("_", " ")}</span><span>${m.ok ? fmtTime(m.t) : "not supported"}</span>`;
  $("inject-log").prepend(li);
  if (m.ok) state.markers.push({ t: m.t, label: m.gesture.replace("_", " ") });
}

// ------------------------------------------------------------------ brainwave bands
const BAND_INFO = {
  delta: ["δ", "Delta"], theta: ["θ", "Theta"], alpha: ["α", "Alpha"],
  beta: ["β", "Beta"], gamma: ["γ", "Gamma"],
};
const bandsState = { key: "", smooth: null, cells: [] };

function buildBands(channels, bands) {
  const el = $("bands");
  el.style.setProperty("--n-ch", channels.length);
  el.innerHTML = "";
  bandsState.cells = [];
  for (const [b, [lo, hi]] of Object.entries(bands)) {     // one row per band
    const [sym, name] = BAND_INFO[b] || ["", b];
    for (const ch of channels) {                            // one column per sensor
      const cell = document.createElement("div");
      cell.className = "band-cell";
      cell.title = `${name} ${lo}–${hi} Hz on ${ch}`;
      cell.innerHTML = `<div class="lbl"><b>${sym} ${name} (${ch})</b><span>–</span></div>
        <div class="band-bar"><i style="background: var(--band-${b})"></i></div>`;
      el.appendChild(cell);
      bandsState.cells.push(cell);
    }
  }
}

function renderBands(m) {
  const names = Object.keys(m.bands), nCh = m.channels.length;
  const key = m.channels.join() + names.join();
  if (key !== bandsState.key) { buildBands(m.channels, m.bands); bandsState.key = key; bandsState.smooth = null; }
  // exponential smoothing so bars glide instead of flickering
  const k = 0.35;
  bandsState.smooth = m.relative.map((row, c) => row.map((v, b) =>
    bandsState.smooth ? bandsState.smooth[c][b] * (1 - k) + v * k : v));
  for (let c = 0; c < nCh; c++) {
    const row = bandsState.smooth[c];
    const top = row.indexOf(Math.max(...row));
    for (let b = 0; b < names.length; b++) {
      const cell = bandsState.cells[b * nCh + c];
      const pct = Math.round(row[b] * 100);
      cell.querySelector(".lbl span").textContent = pct + "%";
      cell.querySelector(".band-bar i").style.width = Math.min(100, row[b] * 100) + "%";
      cell.classList.toggle("dominant", b === top);
    }
  }
}

// ------------------------------------------------------------------ sound (Web Audio, generated locally)
const sound = { on: false, ctx: null };
try { sound.on = localStorage.getItem("bayn.sound") === "on"; } catch (e) { /* storage unavailable */ }

function beep(freqs = [880, 1320], dur = 0.14) {
  if (!sound.on) return;
  try {
    sound.ctx = sound.ctx || new (window.AudioContext || window.webkitAudioContext)();
    if (sound.ctx.state === "suspended") sound.ctx.resume();
    let t = sound.ctx.currentTime;
    for (const f of freqs) {
      const o = sound.ctx.createOscillator(), g = sound.ctx.createGain();
      o.frequency.value = f; o.type = "sine";
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(0.25, t + 0.01);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      o.connect(g).connect(sound.ctx.destination);
      o.start(t); o.stop(t + dur + 0.02);
      t += dur + 0.03;
    }
  } catch (e) { /* audio not available */ }
}

function renderSoundBtn() {
  const b = $("btn-sound");
  b.textContent = sound.on ? "🔔 Sound on" : "🔇 Sound off";
  b.setAttribute("aria-pressed", String(sound.on));
}

// ------------------------------------------------------------------ trigger status + meters
const STATE_TEXT = { armed: "Armed", cooldown: "Cooldown", calibrating: "Calibrating", manual: "Manual only",
  training: "Recording" };

function renderTriggerStatus(s) {
  const badge = $("trigger-state");
  badge.className = "badge " + (STATE_TEXT[s.state] ? s.state : "muted");
  badge.textContent = s.state === "cooldown" ? `Cooldown ${Math.ceil(s.cooldown_remaining_s)} s` : (STATE_TEXT[s.state] || s.state);
  const sel = $("gesture");
  if (document.activeElement !== sel && sel.value !== s.gesture) sel.value = s.gesture;

  const meters = $("meters");
  const key = s.detectors.map((d) => d.name).join();
  if (meters.dataset.key !== key) {
    meters.dataset.key = key;
    meters.innerHTML = s.detectors.map((d) => `
      <div class="meter" data-name="${d.name}">
        <div class="lbl"><b>${d.label}</b><span></span></div>
        <div class="track"><div class="fill"></div><div class="thr" title="threshold"></div></div>
      </div>`).join("");
  }
  for (const d of s.detectors) {
    const el = meters.querySelector(`[data-name="${d.name}"]`);
    if (!el || d.threshold == null) continue;
    const full = d.threshold * 2;                       // threshold sits in the middle of the bar
    el.querySelector(".fill").style.width = Math.min(100, (100 * (d.level || 0)) / full) + "%";
    el.classList.toggle("active", Boolean(d.active));
    el.querySelector(".lbl span").textContent =
      `${(d.level || 0).toFixed(0)} / ${d.threshold.toFixed(0)} ${d.unit}${d.calibrated ? " · calibrated" : ""}`;
  }
  let hint = s.gesture === "manual" ? "Gestures off. Use the button or press T."
    : s.calibrated ? `Calibrated (${s.profile}).` : "Not calibrated yet. Using default thresholds. Calibrate for best results.";
  const blink = s.detectors.find((d) => d.name === "triple_blink");
  $("blink-row").hidden = !blink;
  if (blink) {
    const sel = $("blink-method");
    if (document.activeElement !== sel && s.blink_method && sel.value !== s.blink_method) sel.value = s.blink_method;
    hint += blink.method === "ml"
      ? ` Triple blink: ML classifier${blink.last_p != null ? ` (last P = ${blink.last_p})` : ""}.`
      : blink.wanted === "rule" ? ""
      : blink.model ? " Triple blink: threshold rule (the trained model hasn't beaten it yet; record another session)."
      : " Triple blink: threshold rule. Train the ML detector for fewer false triggers.";
  }
  if (s.training && $("train-overlay").hidden) renderTraining(s.training);     // page reloaded mid-session
  const sz = $("seizure-enabled");
  if (document.activeElement !== sz && s.seizure_enabled !== undefined) sz.checked = Boolean(s.seizure_enabled);
  $("trigger-hint").textContent = hint + (s.suppressed ? ` · ${s.suppressed} gesture(s) ignored during cooldown` : "");
}

// ------------------------------------------------------------------ blink-detector training overlay
let trainTimer = null;
const PROTOCOL_NAMES = { standard: "Standard", noisy: "Noisy", boundaries: "Boundaries" };
async function loadTrainSessions() {
  try {
    const r = await (await fetch("/api/events/training/sessions")).json();
    $("train-sessions-count").textContent = `(${r.sessions.length})`;
    $("train-sessions").innerHTML = r.sessions.map((s) => `<li><span>${PROTOCOL_NAMES[s.protocol] || s.protocol}
      · ${s.condition} · ${s.minutes} min · ${new Date(s.created * 1000).toLocaleString()}</span>
      <button class="btn ghost small-btn" data-delete-session="${escapeHtml(s.id)}" type="button">Delete</button></li>`).join("")
      || '<li class="muted">None yet</li>';
  } catch (e) { /* events service not up yet */ }
}
$("train-sessions").addEventListener("click", async (e) => {
  const id = e.target.dataset.deleteSession;
  if (!id || !confirm("Delete this recording session? This can't be undone.")) return;
  const r = await fetch("/api/events/training/delete", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id }) });
  toast(r.ok ? "Session deleted. Record it again, then Retrain." : "Could not delete the session");
  loadTrainSessions();
});
$("train-condition").addEventListener("change", (e) => { if (e.target.value === "walking") $("train-speak").checked = true; });

// spoken prompts (browser speech, offline voices of the OS) - needed when you can't watch the screen
let lastSpoken = "";
function speak(text) {
  if (!$("train-speak").checked || !("speechSynthesis" in window) || !text || text === lastSpoken) return;
  lastSpoken = text;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text.replace(" - now!", ", now!"));
  u.rate = 1.05;
  speechSynthesis.speak(u);
}

function showTrainIntro() {
  loadTrainSessions();
  $("train-overlay").hidden = false;
  $("train-intro").hidden = false;
  $("train-prompt").hidden = true;
  $("train-progress-bar").hidden = true;
  $("train-cancel").hidden = true;
  $("train-close").hidden = false;
  $("train-count").textContent = "";
  $("train-step").textContent = "Train the triple-blink detector";
}

const pct = (x) => (x == null ? "–" : Math.round(x * 100) + " %");
function trainReport(r) {
  const row = (name, m) => `<tr><td>${name}</td><td>${m.caught}/${m.prompts} (${pct(m.recall)})</td>
    <td>${m.false_triggers} (${m.false_per_min ?? "–"}/min)</td></tr>`;
  const dec = r.decision || {};
  const better = Boolean(dec.ml_clearly_better);
  const tp = r.tp_observation || {};
  const tpRows = Object.entries(tp).map(([c, v]) =>
    `<tr><td>${c.replace("_", " ")}</td><td>${v.n}</td><td>${v.tp_corr.toFixed(2)}</td><td>${v.tp_at_peak.toFixed(2)}×</td></tr>`).join("");
  return `<div class="row"><b>${r.sessions.length} session(s), ${r.minutes} min</b> · examples:
      ${Object.entries(r.class_counts).map(([k, v]) => `${k.replace("_", " ")} ${v}`).join(", ")}</div>
    <div class="row"><b>Held-out test</b> <span class="muted">(${r.cv})</span>
      <table class="train-table"><tr><th></th><th>Triple blinks caught</th><th>False triggers</th></tr>
        ${r.rule_old ? row("Rule, forehead (before)", r.rule_old) : ""}${row("Rule, your blink pattern", r.rule)}${r.ml_real ? row("ML, real data only", r.ml_real) : ""}${row("ML, real + synthetic", r.ml)}</table>
      <b>Is ML necessary?</b> ${dec.answer ? escapeHtml(dec.answer) : "–"}
      <span class="muted">(${dec.final ? "final" : "preliminary, still missing: " + escapeHtml((dec.missing_protocols || []).join("; "))})</span><br>
      ${better ? "✅ Auto uses the ML classifier." : "Auto uses the rule on your learned blink pattern."}
      <span class="muted">Full report: <code>.\\bayn blink decide</code> → docs/BLINK_DECISION.md</span></div>
    <div class="row"><b>TP9/TP10 vs forehead</b> <span class="muted">(median correlation, TP size at the forehead peak)</span>
      <table class="train-table"><tr><th>class</th><th>n</th><th>corr</th><th>TP at peak</th></tr>${tpRows}</table></div>`;
}

function renderTraining(m) {
  const ov = $("train-overlay");
  if (m.state === "cancelled") { ov.hidden = true; return; }
  ov.hidden = false;
  $("train-intro").hidden = true;
  const prompt = $("train-prompt");
  prompt.hidden = false;
  prompt.textContent = m.prompt || "";
  prompt.classList.toggle("do", m.step === "do");
  prompt.classList.toggle("pause", m.step === "pause");
  const running = m.state === "running";
  $("train-progress-bar").hidden = !running && m.state !== "saved";
  $("train-progress").style.width = ((m.progress || 0) * 100) + "%";
  $("train-cancel").hidden = !running;
  $("train-close").hidden = running || m.state === "saved";
  $("train-step").textContent = running
    ? `${PROTOCOL_NAMES[m.protocol] || "Standard"} session (${m.condition || "seated"}) · prompt ${m.prompts_done} of ${m.prompts_total}`
    : m.state === "saved" ? `Session saved (${m.sessions} in total)` : "Train the triple-blink detector";
  if (running && m.step === "do") beep([660], 0.15);
  if (running && m.step !== "rest") speak(m.prompt);
  clearInterval(trainTimer);
  if (running && m.step_ends_at) {
    const tick = () => { $("train-count").textContent = Math.max(0, m.step_ends_at - Date.now() / 1000).toFixed(0) + " s"; };
    tick(); trainTimer = setInterval(tick, 250);
  } else {
    $("train-count").textContent = "";
  }
  $("train-result").innerHTML = m.state === "trained" ? trainReport(m.report)
    : m.state === "train_failed" ? `<div class="row">⚠️ ${escapeHtml(m.prompt)}</div>` : "";
}

// ------------------------------------------------------------------ events: banner + timeline
const events = new Map();      // id -> meta
const SOURCE_TEXT = { gesture: "", button: " (button)", hotkey: " (T key)" };

function showBanner(m) {
  const el = $("event-banner");
  el.textContent = m.kind === "seizure"
    ? `⚠ ${m.label}: mock model, not a diagnosis. Saving EEG −${m.pre_s} s / +${m.post_s} s`
    : `⚡ Event: ${m.label}${SOURCE_TEXT[m.source] || ""}. Saving EEG −${m.pre_s} s / +${m.post_s} s`;
  el.classList.toggle("seizure", m.kind === "seizure");
  el.classList.add("show");
  clearTimeout(showBanner._t);
  showBanner._t = setTimeout(() => el.classList.remove("show"), 4000);
}

function renderTimeline() {
  const list = [...events.values()].sort((a, b) => b.t - a.t);
  $("event-count").textContent = list.length;
  $("timeline-empty").style.display = list.length ? "none" : "";
  const now = Date.now() / 1000;
  $("timeline").innerHTML = list.map((m) => {
    const saved = m.state === "saved";
    const pct = saved ? 100 : Math.min(100, (100 * (now - m.t)) / m.post_s);
    const tr = m.transcript;
    const links = (saved && m.files
      ? `<a href="/api/events/${m.id}/eeg.csv" download>EEG .csv</a> <a href="/api/events/${m.id}/eeg.npz" download>.npz</a>` : "")
      + (tr ? ` <a href="/api/events/${m.id}/transcript.txt" download>🗣 transcript (${tr.n_segments})</a>` : "")
      + (m.summary ? ` <a href="/api/events/${m.id}/summary.md" download>📝 summary</a>` : "");
    const detail = !m.info ? "" : m.info.gaps_s ? ` · gaps ${m.info.gaps_s.join(" / ")} s` : m.info.p != null ? ` · ML P = ${m.info.p}` : m.info.gap_s ? ` · gap ${m.info.gap_s}s` : "";
    return `<li>
      <div class="tl-head"><b>${m.label}${SOURCE_TEXT[m.source] || ""}</b><time>${fmtTime(m.t)}</time></div>
      <div class="tl-meta">
        <span class="chip ${saved ? "saved" : "recording"}">${saved ? `saved ${m.window_s}s${m.partial ? " (partial)" : ""}` : "recording…"}</span>
        <span>conf ${(m.confidence * 100).toFixed(0)}%${detail}</span> ${links}
      </div>
      ${saved ? "" : `<div class="progress"><i style="width:${pct}%"></i></div>`}
    </li>`;
  }).join("");
}
setInterval(() => { if ([...events.values()].some((m) => m.state !== "saved")) renderTimeline(); }, 1000);

function onEvent(m) {
  events.set(m.id, m);
  state.markers.push({ t: m.t, label: "EVENT · " + m.label });
  showBanner(m);
  beep();
  renderTimeline();
}

function onEventDone(m) {
  events.set(m.id, m);
  renderTimeline();
}

// ------------------------------------------------------------------ calibration overlay
let calTimer = null;
function renderCalibration(m) {
  const ov = $("cal-overlay");
  if (m.state === "cancelled") { ov.hidden = true; return; }
  ov.hidden = false;
  const prompt = $("cal-prompt");
  prompt.textContent = m.prompt || "";
  prompt.classList.toggle("do", m.step === "do");
  $("cal-progress").style.width = ((m.progress || 0) * 100) + "%";
  const running = m.state === "running";
  $("cal-cancel").hidden = !running;
  $("cal-close").hidden = running;
  const label = { clench: "Jaw clench", triple_blink: "Triple blink" };
  $("cal-step").textContent = running
    ? (m.step === "rest" ? "Step 1: resting level" : `${label[m.gesture] || ""}: ${m.rep} of ${m.reps}`)
    : "Calibration";
  if (m.step === "do") beep([660], 0.18);
  clearInterval(calTimer);
  if (running && m.step_ends_at) {
    const tick = () => { $("cal-count").textContent = Math.max(0, m.step_ends_at - Date.now() / 1000).toFixed(0) + " s"; };
    tick(); calTimer = setInterval(tick, 250);
  } else {
    $("cal-count").textContent = "";
  }
  const res = $("cal-result");
  res.innerHTML = m.results ? Object.values(m.results).map((r) => `
    <div class="row"><b>${r.label}</b>: ${r.quality === "good" ? "✅" : r.quality === "weak" ? "⚠️" : "❌"} ${r.message}<br>
      <span class="muted">rest ${r.rest_level.toFixed(1)} · gesture ${r.gesture_level.toFixed(1)} · threshold ${r.threshold.toFixed(1)} µV (×${r.separation.toFixed(1)})</span></div>`).join("") : "";
}

// ------------------------------------------------------------------ transcript
const transcript = { ids: new Set() };
const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function inAnyEvent(t) {
  for (const e of events.values()) if (t >= e.t - e.pre_s && t <= e.t + e.post_s) return true;
  return false;
}

function addSegment(s) {
  if (transcript.ids.has(s.id)) return;
  transcript.ids.add(s.id);
  const list = $("transcript");
  const stick = list.scrollHeight - list.scrollTop - list.clientHeight < 40;     // auto-scroll if at bottom
  const li = document.createElement("li");
  li.className = inAnyEvent(s.t_start) ? "in-event" : "";
  li.innerHTML = `<time>${fmtTime(s.t_start)}</time><span class="lang ${s.lang}">${(s.lang || "?").toUpperCase()}</span>
    <span class="text" lang="${s.lang}" dir="auto">${escapeHtml(s.text)}</span>`;
  li.title = `latency ${s.latency_s ?? "?"} s · language confidence ${Math.round((s.lang_prob || 0) * 100)}%`;
  list.appendChild(li);
  while (list.children.length > 200) list.removeChild(list.firstChild);
  if (stick) list.scrollTop = list.scrollHeight;
  $("transcript-empty").style.display = "none";
}

function renderSttStatus(m) {
  const badge = $("stt-model");
  if (m.state === "error") {
    badge.className = "badge bad-badge"; badge.textContent = "transcription off";
    badge.title = m.message; $("partial").textContent = "⚠ " + m.message;
    return;
  }
  if (m.state === "loading") { badge.textContent = "loading model…"; return; }
  badge.className = "badge";
  badge.textContent = `${m.model} · ${m.device === "cuda" ? "GPU" : "CPU"}`;
  badge.title = `compute ${m.compute_type}, beam ${m.beam_size}` +
    (m.last_latency_s != null ? `, last latency ${m.last_latency_s.toFixed(2)} s` : "") +
    (m.fallback ? `\nGPU not used: ${m.fallback}` : "");
  const sel = $("language");
  if (document.activeElement !== sel && sel.value !== m.language) sel.value = m.language;
}

function renderAudioLevel(m) {
  // map -60..0 dBFS to 0..100 %
  $("mic-level").style.width = Math.max(0, Math.min(100, ((m.rms_db + 60) / 60) * 100)) + "%";
}

function renderAudioStatus(m) {
  $("mic-dot").className = "mic-dot " + (m.ok ? "ok" : "bad");
  $("mic-name").textContent = m.ok ? (m.source === "file" ? "file: " : "") + m.device : (m.message || "no audio");
  $("mic-name").title = $("mic-name").textContent;
}

function onTranscriptWindow(m) {
  const e = events.get(m.event_id);
  if (e) { e.transcript = { n_segments: m.n_segments, languages: m.languages }; renderTimeline(); }
  sumEntry(m.event_id).segments = m.segments;
  renderSummaries();
}

// ------------------------------------------------------------------ summaries: heard -> context -> summary
const sums = new Map();   // event_id -> {segments, t_event, langs: {lang: {...}}, active}

// minimal, safe markdown: **bold**, "- " bullets, ### headings, _italic_ (everything else escaped)
function md(text) {
  const out = []; let list = false;
  for (const raw of (text || "").split("\n")) {
    let line = escapeHtml(raw).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|\s)_(.+?)_(\s|$)/g, "$1<em>$2</em>$3");
    if (/^-\s/.test(raw)) { if (!list) { out.push("<ul>"); list = true; } out.push(`<li dir="auto">${line.slice(2)}</li>`); continue; }
    if (list) { out.push("</ul>"); list = false; }
    if (raw.startsWith("### ")) out.push(`<p dir="auto"><b>${line.slice(4)}</b></p>`);
    else if (raw.trim()) out.push(`<p dir="auto">${line}</p>`);
  }
  if (list) out.push("</ul>");
  return out.join("");
}

function sumEntry(id) {
  if (!sums.has(id)) sums.set(id, { segments: null, langs: {}, active: null });
  return sums.get(id);
}

function renderSummaries() {
  const open = new Set([...document.querySelectorAll("#summaries details[open]")].map((d) => d.dataset.key));
  renderSummariesInner();
  document.querySelectorAll("#summaries details").forEach((d) => { if (open.has(d.dataset.key)) d.open = true; });
}

function renderSummariesInner() {
  const ids = [...sums.keys()].filter((id) => events.has(id)).sort((a, b) => events.get(b).t - events.get(a).t).slice(0, 8);
  $("summaries-empty").style.display = ids.length ? "none" : "";
  $("summaries").innerHTML = ids.map((id) => {
    const s = sums.get(id), e = events.get(id);
    const langs = Object.keys(s.langs);
    const lang = s.active && s.langs[s.active] ? s.active : langs[0];
    const cur = s.langs[lang] || {};
    const heard = s.segments == null ? `<p class="muted small">waiting for the transcript…</p>`
      : s.segments.length ? `<ol class="heard">${s.segments.map((g) => `<li dir="auto"><time>${(g.t_start - e.t >= 0 ? "+" : "") + (g.t_start - e.t).toFixed(0)} s</time>${escapeHtml(g.text)}</li>`).join("")}</ol>`
      : `<p class="muted small">no speech in this window</p>`;
    const tabs = langs.length > 1 ? `<span class="tabs">${langs.map((l) => `<button data-sum="${id}" data-lang="${l}" class="${l === lang ? "on" : ""}">${l.toUpperCase()}</button>`).join("")}</span>` : "";
    const prompt = cur.prompt ? `<details data-key="${id}-${lang}"><summary>Prompt sent to the LLM</summary><pre dir="auto">${escapeHtml(cur.prompt.map((m) => `[${m.role}]\n${m.content}`).join("\n\n"))}</pre></details>` : "";
    const body = cur.error ? `<p class="err">⚠ ${escapeHtml(cur.error)}</p>`
      : cur.text ? `<div class="md ${cur.final ? "" : "writing"}">${md(cur.text)}</div>`
      : cur.started ? `<div class="md writing"></div>` : `<p class="muted small">queued…</p>`;
    const meta = cur.final && !cur.error ? `<p class="sum-meta">${cur.model} · ${cur.seconds} s${cur.tokens_per_s ? ` · ${cur.tokens_per_s} tok/s` : ""}</p>` : "";
    return `<article class="sum">
      <div class="sum-head"><b>${escapeHtml(e.label)}${SOURCE_TEXT[e.source] || ""} · ${fmtTime(e.t)}</b>
        <span class="actions">${tabs}
          <button class="btn ghost" data-regen="${id}" data-lang="${lang || ""}" type="button">↻ Regenerate</button>
          ${cur.final ? `<a href="/api/events/${id}/summary.md" download>summary.md</a>` : ""}</span></div>
      <div class="sum-grid">
        <div class="sum-col"><h3>① Heard</h3>${heard}</div>
        <div class="sum-col"><h3>② Context</h3><div class="md">${md(cur.eeg || "")}</div>${prompt}</div>
        <div class="sum-col"><h3>③ Summary${cur.model ? " · " + escapeHtml(cur.model) : ""}</h3>${body}${meta}</div>
      </div></article>`;
  }).join("");
}

function onSummaryToken(m) {
  const s = sumEntry(m.event_id);
  const cur = (s.langs[m.lang] = s.langs[m.lang] || {});
  if (m.started) Object.assign(cur, { started: true, final: false, error: null, prompt: m.prompt, eeg: m.eeg, model: m.model, text: "" });
  if (m.text) cur.text = m.text;
  s.active = s.active || m.lang;
  renderSummaries();
}

function onSummaryDone(m) {
  const s = sumEntry(m.event_id);
  s.langs[m.lang] = { ...s.langs[m.lang], ...m, text: m.llm_text, final: true, started: true };
  s.active = s.active || m.lang;
  const e = events.get(m.event_id);
  if (e) { e.summary = { languages: Object.keys(s.langs) }; renderTimeline(); }
  renderSummaries();
}

async function loadSummaryHistory() {
  // past events: fetch saved transcript + summary files (most recent few)
  const recent = [...events.values()].filter((e) => e.transcript || e.summary).sort((a, b) => b.t - a.t).slice(0, 5);
  for (const e of recent) {
    const s = sumEntry(e.id);
    try {
      const tr = await (await fetch(`/api/events/${e.id}/transcript.json`)).json();
      s.segments = tr.segments;
    } catch (err) { /* not there yet */ }
    try {
      for (const r of await (await fetch(`/api/events/${e.id}/summary.json`)).json())
        s.langs[r.lang] = { ...r, text: r.llm_text, final: true, started: true };
    } catch (err) { /* not summarised yet */ }
  }
  renderSummaries();
}

document.addEventListener("click", (ev) => {
  const t = ev.target.closest("[data-sum],[data-regen]");
  if (!t) return;
  if (t.dataset.sum) { sumEntry(t.dataset.sum).active = t.dataset.lang; renderSummaries(); }
  if (t.dataset.regen) post("/api/summary/regenerate", { event_id: t.dataset.regen, lang: t.dataset.lang || null });
});

function renderSumStatus(m) {
  const b = $("sum-model");
  b.className = "badge " + (m.state === "ready" ? "" : m.state === "error" ? "bad-badge" : "muted");
  b.textContent = m.state === "ready" ? `${m.model} · ready` : m.state === "loading" ? `loading ${m.model || "LLM"}…` : m.state === "error" ? "LLM off" : "LLM…";
  b.title = m.message || (m.last && m.last.seconds ? `last summary ${m.last.seconds} s, ${m.last.tokens_per_s} tok/s` : "Local LLM via Ollama");
  const sel = $("sum-language");
  if (document.activeElement !== sel && sel.value !== m.language) sel.value = m.language;
}

async function post(url, body) {
  try {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) toast((await r.json()).detail || "Request failed");
  } catch (e) { toast("Dashboard server not reachable"); }
}

// ------------------------------------------------------------------ websocket
function connect(delay = 500) {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => setConn("", "Waiting for EEG…");
  ws.onclose = () => { setConn("down", "Dashboard offline"); setTimeout(() => connect(Math.min(delay * 2, 5000)), delay); };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    switch (m.type) {
      case "hello":
        Object.assign(state, { channels: m.channels, fs: m.fs, seconds: m.plot_seconds });
        $("version").textContent = "v" + m.version;
        $("filter-note").textContent = `Display filter: ${m.bandpass[0]}–${m.bandpass[1]} Hz bandpass, notch ${m.notch === "auto" ? "50+60 Hz (auto)" : m.notch === "off" ? "off" : m.notch + " Hz"}`;
        resetBuffers();
        break;
      case "eeg": pushEEG(m); break;
      case "eeg_status": renderStatus(m); break;
      case "injected": logInjection(m); break;
      case "bands": renderBands(m); break;
      case "trigger_status": renderTriggerStatus(m); break;
      case "event": onEvent(m); break;
      case "event_done": onEventDone(m); break;
      case "calibration": renderCalibration(m); break;
      case "training":
        if (m.state === "trained" || m.state === "train_failed") {
          if (!$("train-overlay").hidden) renderTraining(m);       // replayed results don't pop up on reload
          else if (m.state === "trained" && (Date.now() / 1000 - m.t) < 10) toast("Blink detector trained");
        } else renderTraining(m);
        break;
      case "event_history":
        for (const e of m.events) events.set(e.id, e);
        renderTimeline();
        loadSummaryHistory();
        break;
      case "summary_token": onSummaryToken(m); break;
      case "summary": onSummaryDone(m); break;
      case "summary_status": renderSumStatus(m); break;
      case "transcript": addSegment(m); break;
      case "transcript_history": m.segments.forEach(addSegment); break;
      case "transcript_partial": $("partial").textContent = m.done ? "" : (m.text ? "… " + m.text : ""); break;
      case "transcript_status": renderSttStatus(m); break;
      case "transcript_window": onTranscriptWindow(m); break;
      case "audio_level": renderAudioLevel(m); break;
      case "audio_status": renderAudioStatus(m); break;
      case "heartbeat":
        if (m.eeg_age_s !== null && m.eeg_age_s <= 2) setConn("live", "EEG live");
        else setConn("down", state.statusMsg ? capitalize(state.statusMsg)
                             : m.eeg_age_s === null ? "No EEG yet" : "EEG signal lost");
        break;
    }
  };
}

// ------------------------------------------------------------------ controls
$("scale").addEventListener("change", (e) => { state.scale = Number(e.target.value); });
$("btn-pause").addEventListener("click", (e) => {
  state.paused = !state.paused;
  e.target.textContent = state.paused ? "Resume" : "Pause";
});
document.querySelectorAll("[data-inject]").forEach((b) =>
  b.addEventListener("click", async () => {
    const r = await fetch("/api/sim/inject", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gesture: b.dataset.inject }),
    });
    if (!r.ok) toast("Injection failed: " + (await r.json()).detail);
  }));
document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, textarea, select") || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === "t" || e.key === "T") post("/api/events/trigger", { source: "hotkey" });
});
$("btn-trigger").addEventListener("click", () => post("/api/events/trigger", { source: "button" }));
$("gesture").addEventListener("change", (e) => post("/api/events/gesture", { gesture: e.target.value }));
$("language").addEventListener("change", (e) => post("/api/transcription/language", { language: e.target.value }));
$("sum-language").addEventListener("change", (e) => post("/api/summary/language", { language: e.target.value }));
$("btn-calibrate").addEventListener("click", () => post("/api/events/calibration", { action: "start" }));
$("cal-cancel").addEventListener("click", () => post("/api/events/calibration", { action: "cancel" }));
$("cal-close").addEventListener("click", () => { $("cal-overlay").hidden = true; });
$("btn-train").addEventListener("click", showTrainIntro);
document.querySelectorAll("[data-protocol]").forEach((b) => b.addEventListener("click",
  () => post("/api/events/training", { action: "start", protocol: b.dataset.protocol,
                                       condition: $("train-condition").value })));
$("train-retrain").addEventListener("click", () => {
  post("/api/events/training", { action: "retrain" });
  renderTraining({ state: "saved", prompt: "Training the detector on the saved sessions...", sessions: "all" });
});
$("train-cancel").addEventListener("click", () => post("/api/events/training", { action: "cancel" }));
$("train-close").addEventListener("click", () => { $("train-overlay").hidden = true; });
$("blink-method").addEventListener("change", (e) => post("/api/events/blink_method", { method: e.target.value }));
$("seizure-enabled").addEventListener("change", (e) => post("/api/events/seizure", { enabled: e.target.checked }));
$("btn-sound").addEventListener("click", () => {
  sound.on = !sound.on;
  try { localStorage.setItem("bayn.sound", sound.on ? "on" : "off"); } catch (e) { /* ignore */ }
  renderSoundBtn();
  if (sound.on) beep([880], 0.1);                  // also unlocks audio (needs a user click)
});
renderSoundBtn();

resetBuffers();
connect();
requestAnimationFrame(draw);
