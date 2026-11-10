"""A recorded, install-free preview of the dashboard (for judges and the README).

    python -m bayn.preview record   # while `bayn showcase` runs: saves the dashboard's live stream
                                    # (data/preview/session.json) and the events' files
    python -m bayn.preview export   # docs/preview/: a static copy of the dashboard that replays the
                                    # recording in any browser (open index.html, or host it, e.g. GitHub Pages)
    python -m bayn.preview gif      # docs/media/*.gif from that replay (headless Edge/Chrome + Pillow; dev tool)

The replay is the real dashboard code (src/bayn/ui/static) fed by a recorded WebSocket stream (replay.js):
same plots, transcript, events and streamed summaries. Controls are inert (a toast says how to run it).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

from bayn.common.config import DATA_DIR, ROOT

STATIC = ROOT / "src" / "bayn" / "ui" / "static"
SESSION = DATA_DIR / "preview" / "session.json"
OUT = ROOT / "docs" / "preview"
EVENT_FILES = ("transcript.txt", "transcript.json", "summary.md", "summary.json", "eeg.csv")
SKIP_AT_CONNECT = {"training", "calibration"}        # leftovers from earlier sessions, replayed on connect


def compact(m: dict) -> dict:
    """Smaller recording: EEG to 0.1 µV (older servers sent float32 noise digits), other floats to 4 digits."""
    def r(v, nd):
        if isinstance(v, float):
            return round(v, nd)
        if isinstance(v, list):
            return [r(x, nd) for x in v]
        if isinstance(v, dict):
            return {k: (v[k] if k in ("t", "t0", "created", "recorded") else r(v[k], nd)) for k in v}
        return v
    return r(m, 1 if m.get("type") == "eeg" else 4)


def trim(s: dict, seconds: float) -> dict:
    """Keep the first `seconds` of a recording (and only the events that started in it)."""
    s["messages"] = [(t, m) for t, m in s["messages"] if t <= seconds]
    ids = {m["id"] for _, m in s["messages"] if m["type"] == "event"}
    s["events"] = [e for e in s["events"] if e in ids]
    s["seconds"] = seconds
    return s


def record(url: str = "ws://127.0.0.1:8080/ws", seconds: float = 260.0, out: Path = SESSION,
           wait_for_scene: bool = True) -> Path:
    """Record the dashboard stream. With wait_for_scene, recording starts when the showcase (re)starts at
    its intro, so the replay is one clean loop."""
    from websockets.sync.client import connect

    msgs: list[tuple[float, dict]] = []
    events: list[str] = []
    with connect(url, max_size=None) as ws:
        hello = []
        t_start = None
        start_pos = 0.0
        while True:
            m = json.loads(ws.recv())
            if t_start is None:
                if m["type"] in ("hello", "trigger_status", "summary_status", "transcript_status", "audio_status",
                                 "eeg_status"):
                    hello = [x for x in hello if x["type"] != m["type"]] + [m]     # latest state of each
                if not wait_for_scene or (m["type"] == "showcase" and m.get("index") == 0):
                    t_start = time.time()
                    start_pos = float(m.get("pos", 0.0)) if m["type"] == "showcase" else 0.0
                    print(f"recording... (tour position {start_pos:.1f} s)")
                    msgs += [(0.0, x) for x in hello]
                    msgs += [(0.0, {"type": "event_history", "events": []}),
                             (0.0, {"type": "transcript_history", "segments": []})]
                    if m["type"] not in [x["type"] for x in hello]:
                        msgs.append((0.0, m))
                continue
            t = time.time() - t_start
            if m["type"] in SKIP_AT_CONNECT or m["type"] in ("event_history", "transcript_history"):
                continue
            if m["type"] == "event" and m["id"] not in events:
                events.append(m["id"])
            msgs.append((round(t, 3), compact(m)))
            if t >= seconds:
                break
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"recorded": t_start, "seconds": seconds, "start_pos": start_pos, "events": events,
                               "messages": msgs}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(msgs)} messages, {len(events)} events -> {out}")
    return out


def export(session: Path = SESSION, out: Path = OUT) -> Path:
    """Static dashboard + replay. Event files (transcripts, summaries, EEG csv) are copied so the download
    links work too."""
    from bayn.events.store import EventStore

    s = json.loads(session.read_text(encoding="utf-8"))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("styles.css", "favicon.png", "logo-mark.png"):
        shutil.copy2(STATIC / name, out / name)
    html = (STATIC / "index.html").read_text(encoding="utf-8").replace("/static/", "")
    html = html.replace('<script src="app.js"></script>', '<script src="session.js"></script>\n  '
                        '<script src="replay.js"></script>\n  <script src="app.js"></script>')
    html = html.replace("<title>Bayn Dashboard</title>", "<title>Bayn Dashboard · recorded preview</title>")
    (out / "index.html").write_text(html, encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    (out / "app.js").write_text(re.sub(r'href="/api/', 'href="api/', js), encoding="utf-8")
    shutil.copy2(Path(__file__).parent / "ui" / "replay.js", out / "replay.js")
    store = EventStore()
    files = {}                        # what the page fetches, inlined, so it also works opened from disk
    for eid in s["events"]:
        src, dst = store.dir_for(eid), out / "api" / "events" / eid
        dst.mkdir(parents=True)
        for name in EVENT_FILES:      # ...and as files, for the download links
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
                if name.endswith(".json"):
                    files[f"api/events/{eid}/{name}"] = json.loads((src / name).read_text(encoding="utf-8"))
    s["files"] = files
    body = json.dumps(s, ensure_ascii=False, separators=(",", ":"))
    (out / "session.js").write_text(f"window.BAYN_SESSION = {body};\n", encoding="utf-8")   # a script, not a
    # fetch: browsers block fetch() for pages opened from disk (file://)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"preview: {out} ({size:.1f} MB, {len(s['events'])} events)")
    return out


# ---------------------------------------------------------------- GIFs from the replay (dev tool)
MEDIA = ROOT / "docs" / "media"
BROWSERS = [Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
            Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
            Path("C:/Program Files/Google/Chrome/Application/chrome.exe")]


class Browser:
    """Headless Edge/Chrome over the DevTools protocol (no extra dependency: `websockets` is already installed)."""

    def __init__(self, width: int, height: int) -> None:
        import socket
        import subprocess
        import tempfile
        import urllib.request

        exe = next((b for b in BROWSERS if b.exists()), None) or shutil.which("msedge") or shutil.which("chrome")
        if not exe:
            raise SystemExit("needs Microsoft Edge or Google Chrome")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.profile = tempfile.mkdtemp(prefix="bayn-gif-")
        self.proc = subprocess.Popen([str(exe), "--headless=new", f"--remote-debugging-port={port}",
                                      f"--user-data-dir={self.profile}", f"--window-size={width},{height}",
                                      "--hide-scrollbars", "--force-device-scale-factor=1", "--mute-audio",
                                      "--no-first-run", "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        url = None
        for _ in range(100):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=1).read())
                url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        if url is None:
            self.proc.terminate()
            raise SystemExit("browser did not start")
        from websockets.sync.client import connect

        self.ws = connect(url, max_size=None)
        self.n = 0
        self.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)

    def call(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.n:
                if "error" in m:
                    raise RuntimeError(m["error"])
                return m.get("result", {})

    def js(self, expr: str):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r["result"].get("value")

    def shot(self, clip: dict | None = None):
        import base64
        import io

        from PIL import Image

        kw: dict = {"format": "png"}
        if clip:
            kw.update(clip={**clip, "scale": 1}, captureBeyondViewport=True)
        data = base64.b64decode(self.call("Page.captureScreenshot", **kw)["data"])
        return Image.open(io.BytesIO(data)).convert("RGB")

    def close(self) -> None:
        try:
            self.ws.close()
        finally:
            self.proc.terminate()
            shutil.rmtree(self.profile, ignore_errors=True)


def save_gif(frames: list, path: Path, frame_ms: int, width: int | None = None) -> float:
    from PIL import Image

    if width:
        frames = [f.resize((width, round(f.height * width / f.width)), Image.LANCZOS) for f in frames]
    pal = frames[len(frames) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(path, save_all=True, append_images=q[1:], duration=frame_ms, loop=0, optimize=True)
    return path.stat().st_size / 1e6


def capture(browser: Browser, base: str, name: str, start: float, end: float, speed: float, every_s: float,
            region: str | None, frame_ms: int, width: int | None, scale: int = 50, hold_s: float = 1.5) -> None:
    """Replay [start, end] of the recording at `speed`, one frame every `every_s` of real time. `region` = a CSS
    selector to crop to ("selector@px": fixed height, for cards that grow), scrolled to just below the sticky
    header; `scale` = the EEG plot's range (µV)."""
    browser.call("Page.navigate", url=f"{base}?from={start}&to={end}&speed={speed}&ribbon=0")
    deadline = time.time() + 10
    while not browser.js("document.readyState === 'complete'") and time.time() < deadline:
        time.sleep(0.1)
    browser.js(f"(() => {{ const s = document.getElementById('scale'); s.value = '{scale}';"
               f" s.dispatchEvent(new Event('change')); }})()")
    time.sleep(0.6)
    clip = None
    if region:
        css, _, px = region.partition("@")
        sel = json.dumps(css)
        r = browser.js(f"""(() => {{
            const el = document.querySelector({sel}), bar = document.querySelector('.topbar');
            scrollTo(0, Math.max(0, el.getBoundingClientRect().top + scrollY - bar.offsetHeight - 8));
            const b = el.getBoundingClientRect();
            return {{x: b.x + scrollX, y: b.y + scrollY, width: b.width, height: {px or 'b.height'}}}; }})()""")
        clip = {k: max(0, round(v)) for k, v in r.items()}       # page coordinates
        time.sleep(0.2)
    frames = []
    while not browser.js("!!window.replayDone"):
        t = time.time()
        frames.append(browser.shot(clip))
        time.sleep(max(0.0, every_s - (time.time() - t)))
    frames.append(browser.shot(clip))
    frames += [frames[-1]] * max(0, round(hold_s * 1000 / frame_ms))     # rest on the last frame
    size = save_gif(frames, MEDIA / f"{name}.gif", frame_ms, width)
    print(f"  {name}.gif: {len(frames)} frames, {size:.1f} MB")


def gifs(preview: Path = OUT, only: list[str] | None = None) -> None:
    """The README GIFs, from the exported preview. Scene times come from the showcase timeline."""
    import functools
    import http.server
    import threading

    from bayn.showcase import OUT_WAV

    tl = json.loads(OUT_WAV.with_suffix(".json").read_text(encoding="utf-8"))
    sc = {s["trigger"]: s for s in tl["scenes"]}

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a) -> None:
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(preview)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/index.html"
    cl, sz = sc["clench"], sc["seizure"]
    # (no triple-blink GIF: docs/media/triple-blink.gif is the team's own recording on a real headset)
    shots = {   # name: (start, end, speed, every_s, region, frame_ms, width, EEG scale µV)
        "dashboard": (cl["trigger_s"] - 6, cl["trigger_s"] + 6, 1.0, 0.25, None, 250, 1100, 50),
        "transcript": (cl["audio_s"], cl["audio_end_s"], 3.0, 0.3, ".transcript-card@470", 300, 760, 50),
        "summary": (cl["trigger_s"] + 28, cl["trigger_s"] + 60, 2.0, 0.3, ".summary-card@250", 300, 900, 50),
        "seizure": (sz["trigger_s"] + 8, sz["trigger_s"] + 45, 3.0, 0.3, None, 300, 1100, 100),
    }
    off = json.loads(SESSION.read_text(encoding="utf-8")).get("start_pos", 0.0)
    browser = Browser(1400, 900)
    try:
        for name, (a, b, speed, every, region, ms, w, scale) in shots.items():
            if not only or name in only:      # tour position -> time in the recording
                capture(browser, base, name, max(0.0, a - off), b - off, speed, every, region, ms, w, scale)
    finally:
        browser.close()
        srv.shutdown()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["record", "export", "gif"])
    ap.add_argument("names", nargs="*", help="gif: only these (dashboard transcript summary seizure)")
    ap.add_argument("--seconds", type=float, help="record: length (default: one loop of the showcase)")
    ap.add_argument("--url", default="ws://127.0.0.1:8080/ws")
    ap.add_argument("--now", action="store_true", help="record: start immediately (don't wait for the intro)")
    args = ap.parse_args(argv)
    if args.cmd == "record":
        if args.seconds is None:                       # one loop of the tour
            from bayn.showcase import OUT_WAV

            args.seconds = json.loads(OUT_WAV.with_suffix(".json").read_text(encoding="utf-8"))["length_s"] - 1
        record(args.url, args.seconds, wait_for_scene=not args.now)
    elif args.cmd == "export":
        export()
    else:
        gifs(only=args.names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
