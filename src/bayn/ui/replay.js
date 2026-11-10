// Recorded preview of the Bayn dashboard (python -m bayn.preview export).
// Loaded before app.js: replaces the dashboard's WebSocket with a replay of session.json, serves the
// recorded event files for its fetches, and turns the controls into a hint. The clock follows the recording,
// so times and progress bars look as they did live. Loops at the end.
// URL options (GIF capture, python -m bayn.preview gif): ?from=s (jump there), ?to=s (stop), ?speed=x, ?ribbon=0
(() => {
  const q = new URLSearchParams(location.search);
  const FROM = Number(q.get("from") || 0), TO = q.has("to") ? Number(q.get("to")) : Infinity;
  const SPEED = Number(q.get("speed") || 1);
  const session = Promise.resolve(window.BAYN_SESSION);    // session.js: a script, so it works from disk too
  const realFetch = window.fetch.bind(window);
  const realNow = Date.now.bind(Date);
  let clock = null;                       // {rec: recorded unix ms at replay start, wall: real ms at replay start}
  Date.now = () => (clock ? clock.rec + SPEED * (realNow() - clock.wall) : realNow());

  const json = (body, status = 200) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  window.fetch = async (url, opts = {}) => {
    const u = String(url);
    if (!u.includes("api/")) return realFetch(url, opts);
    if ((opts.method || "GET").toUpperCase() !== "GET")
      return json({ detail: "This is a recorded preview. Run Bayn yourself to use the controls (see the README)." }, 409);
    if (u.includes("/training/sessions")) return json({ sessions: [] });
    const f = (window.BAYN_SESSION.files || {})[u.replace(/^\//, "")];   // recorded event files
    return f === undefined ? json({ detail: "not in this recording" }, 404) : json(f);
  };

  class ReplaySocket {
    constructor() {
      this.readyState = 0;
      session.then((s) => this._play(s));
    }
    _play(s) {
      this.readyState = 1;
      if (this.onopen) this.onopen();
      const msgs = s.messages, wall = realNow();
      clock = { rec: (s.recorded + FROM) * 1000, wall };
      let i = 0;
      const step = () => {
        if (this.readyState !== 1) return;
        const t = FROM + SPEED * (realNow() - wall) / 1000;
        while (i < msgs.length && msgs[i][0] <= Math.min(t, TO)) {
          if (this.onmessage) this.onmessage({ data: JSON.stringify(msgs[i][1]) });
          i++;
        }
        if (t >= TO) { window.replayDone = true; return; }
        if (i >= msgs.length) { setTimeout(() => location.reload(), 4000); return; }   // loop
        setTimeout(step, 40);
      };
      step();
    }
    send() {}
    close() { this.readyState = 3; }
  }
  window.WebSocket = ReplaySocket;

  // a ribbon saying what this is
  document.addEventListener("DOMContentLoaded", () => {
    if (q.get("ribbon") === "0") return;
    const bar = document.createElement("div");
    bar.className = "preview-ribbon";
    bar.innerHTML = "Recorded preview: <b>bayn showcase</b> (simulated EEG, recorded speech, real transcripts and " +
      "summaries from a local model). Controls are inactive here. " +
      "<a href=\"https://github.com/Omar0Rawas/bayn#try-it-judges\">Run it yourself</a>";
    document.body.prepend(bar);
    const st = document.createElement("style");
    st.textContent = ".preview-ribbon{background:var(--accent);color:var(--accent-ink);padding:6px 20px;font-size:13px;text-align:center}" +
      ".preview-ribbon a{color:inherit;font-weight:600;margin-left:6px}";
    document.head.append(st);
  });
})();
