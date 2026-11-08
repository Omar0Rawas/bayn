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
};

// ------------------------------------------------------------------ helpers
function toast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove("show"), 2200);
}
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
      case "heartbeat":
        if (m.eeg_age_s === null) setConn("down", "No EEG yet");
        else if (m.eeg_age_s > 2) setConn("down", "EEG signal lost");
        else setConn("live", "EEG live");
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
  if (e.target.matches("input, textarea, select")) return;
  if (e.key === "t" || e.key === "T") toast("Manual trigger arrives in Phase 3");
});

resetBuffers();
connect();
requestAnimationFrame(draw);
