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
const STATE_TEXT = { armed: "Armed", cooldown: "Cooldown", calibrating: "Calibrating", manual: "Manual only" };

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
  const hint = s.gesture === "manual" ? "Gestures off. Use the button or press T."
    : s.calibrated ? `Calibrated (${s.profile}).` : "Not calibrated yet. Using default thresholds. Calibrate for best results.";
  $("trigger-hint").textContent = hint + (s.suppressed ? ` · ${s.suppressed} gesture(s) ignored during cooldown` : "");
}

// ------------------------------------------------------------------ events: banner + timeline
const events = new Map();      // id -> meta
const SOURCE_TEXT = { gesture: "", button: " (button)", hotkey: " (T key)" };

function showBanner(m) {
  const el = $("event-banner");
  el.textContent = `⚡ Event: ${m.label}${SOURCE_TEXT[m.source] || ""}. Saving EEG −${m.pre_s} s / +${m.post_s} s`;
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
    const links = saved && m.files
      ? `<a href="/api/events/${m.id}/eeg.csv" download>EEG .csv</a> <a href="/api/events/${m.id}/eeg.npz" download>.npz</a>` : "";
    const detail = m.info && m.info.gap_s ? ` · gap ${m.info.gap_s}s` : "";
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
  const label = { clench: "Jaw clench", double_blink: "Double blink" };
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
      case "event_history":
        for (const e of m.events) events.set(e.id, e);
        renderTimeline();
        break;
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
$("btn-calibrate").addEventListener("click", () => post("/api/events/calibration", { action: "start" }));
$("cal-cancel").addEventListener("click", () => post("/api/events/calibration", { action: "cancel" }));
$("cal-close").addEventListener("click", () => { $("cal-overlay").hidden = true; });
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
