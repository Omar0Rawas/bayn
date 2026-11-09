"""System test: the real bridge and dashboard processes, end to end, with simulated EEG.

Exercises: process startup, ZeroMQ between processes, filtering, quality + mains
detection, WebSocket relay, the REST injection command and its round trip.
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.system


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http(url: str, timeout: float = 30.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(url)


TINY_READY = (Path(__file__).parents[1] / "models" / "whisper" / "tiny" / "model.bin").exists()


@pytest.fixture
def running_stack(tmp_path):
    from tests.fake_ollama import FakeOllama

    ollama = FakeOllama()                       # deterministic stand-in for the local LLM
    ui, pub, cmd, epub, ecmd, apub, spub, scmd, mpub, mcmd = (_free_port() for _ in range(10))
    env = dict(os.environ, BAYN_EEG__SOURCE="sim", BAYN_UI__PORT=str(ui),
               BAYN_BUS__SUM_PUB_PORT=str(mpub), BAYN_BUS__SUM_CMD_PORT=str(mcmd),
               BAYN_SUMMARY__HOST=ollama.url, BAYN_SUMMARY__MODEL="qwen2.5:3b",
               BAYN_BUS__EEG_PUB_PORT=str(pub), BAYN_BUS__EEG_CMD_PORT=str(cmd),
               BAYN_BUS__EVENTS_PUB_PORT=str(epub), BAYN_BUS__EVENTS_CMD_PORT=str(ecmd),
               BAYN_BUS__AUDIO_PUB_PORT=str(apub), BAYN_BUS__STT_PUB_PORT=str(spub), BAYN_BUS__STT_CMD_PORT=str(scmd),
               # speech: the English test clip, looping, stands in for the microphone; tiny model on CPU
               BAYN_AUDIO__SOURCE="file", BAYN_AUDIO__FILE=str(Path(__file__).parent / "data" / "en_short.wav"),
               BAYN_TRANSCRIPTION__MODEL="tiny", BAYN_TRANSCRIPTION__DEVICE="cpu",
               BAYN_TRANSCRIPTION__EVENT_GRACE_S="6",
               BAYN_SIMULATOR__SEED="5", BAYN_SIMULATOR__MAINS_HZ="60",
               BAYN_SIMULATOR__NATURAL_BLINKS="false", BAYN_SIMULATOR__NATURAL_EMG="false",
               BAYN_SIMULATOR__AUTO_GESTURES__ENABLED="false", BAYN_LOGGING__LEVEL="WARNING",
               # short windows / calibration so the test runs in seconds
               BAYN_EVENTS__PRE_S="2", BAYN_EVENTS__POST_S="3", BAYN_EVENTS__COOLDOWN_S="3",
               BAYN_EVENTS__CALIBRATION__REST_S="5", BAYN_EVENTS__CALIBRATION__REPS="2",
               BAYN_EVENTS__CALIBRATION__READY_S="1", BAYN_EVENTS__CALIBRATION__DO_S="3",
               BAYN_EVENTS__CALIBRATION__RELEASE_S="1.5",
               BAYN_DATA_DIR=str(tmp_path))
    modules = ["bayn.eeg.bridge", "bayn.events.service", "bayn.ui.server"]
    if TINY_READY:
        modules += ["bayn.transcription.audio", "bayn.transcription.service", "bayn.summarization.service"]
    procs = [subprocess.Popen([sys.executable, "-m", m], env=env) for m in modules]
    try:
        _wait_http(f"http://127.0.0.1:{ui}/api/health")
        yield ui
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=10)
        ollama.close()


def test_end_to_end_simulated_pipeline(running_stack):
    import websockets

    ui = running_stack

    async def scenario() -> dict:
        seen: dict = {"types": {}, "eeg_samples": 0, "status": None, "injected": None, "hello": None,
                      "bands": None}
        async with websockets.connect(f"ws://127.0.0.1:{ui}/ws") as ws:
            injected_sent = False
            end = time.time() + 9          # mains detection needs >= 4 s of data
            while time.time() < end:
                m = json.loads(await asyncio.wait_for(ws.recv(), 5))
                seen["types"][m["type"]] = seen["types"].get(m["type"], 0) + 1
                if m["type"] == "hello":
                    seen["hello"] = m
                elif m["type"] == "eeg":
                    seen["eeg_samples"] += len(m["data"][0])
                    assert len(m["data"]) == 4
                elif m["type"] == "eeg_status":
                    seen["status"] = m
                    if not injected_sent and m["connected"]:
                        req = urllib.request.Request(
                            f"http://127.0.0.1:{ui}/api/sim/inject", method="POST",
                            data=json.dumps({"gesture": "double_blink"}).encode(),
                            headers={"Content-Type": "application/json"})
                        assert urllib.request.urlopen(req).status == 202
                        injected_sent = True
                elif m["type"] == "injected":
                    seen["injected"] = m
                elif m["type"] == "bands":
                    seen["bands"] = m
        return seen

    seen = asyncio.run(scenario())
    assert seen["hello"]["source"] == "sim"
    assert 256 * 6 < seen["eeg_samples"] < 256 * 11          # ~real time, no big drops
    st = seen["status"]
    assert st["connected"] and st["supports_injection"]
    assert [q["channel"] for q in st["quality"]] == ["TP9", "AF7", "AF8", "TP10"]
    assert all(q["level"] in ("good", "ok") for q in st["quality"])
    assert st["mains_hz"] == 60                                 # detected, not assumed
    assert seen["injected"]["ok"] and seen["injected"]["gesture"] == "double_blink"
    b = seen["bands"]                                           # brainwave panel data, ~2 Hz
    assert seen["types"]["bands"] >= 8
    assert list(b["bands"]) == ["delta", "theta", "alpha", "beta", "gamma"]
    assert all(abs(sum(row) - 1) < 0.01 for row in b["relative"])


def _post(ui: int, path: str, body: dict) -> int:
    req = urllib.request.Request(f"http://127.0.0.1:{ui}{path}", method="POST", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req).status


async def _collect(ui: int, until, timeout: float, actions=None) -> list[dict]:
    """Read WebSocket messages until until(msgs) is true; actions(msg) may fire REST calls."""
    import websockets

    msgs: list[dict] = []
    async with websockets.connect(f"ws://127.0.0.1:{ui}/ws", max_size=None) as ws:
        end = time.time() + timeout
        while time.time() < end and not until(msgs):
            m = json.loads(await asyncio.wait_for(ws.recv(), 10))
            if m["type"] not in ("eeg", "bands", "heartbeat"):
                msgs.append(m)
            if actions:
                actions(m)
    return msgs


def test_gesture_event_saved_and_downloadable(running_stack):
    ui = running_stack
    fired, first = {}, {}

    def actions(m):
        # wait for >= 4 s of EEG so the 2 s pre-event window is fully covered
        if m["type"] == "eeg":
            first.setdefault("t", m["t0"])
            first["now"] = m["t0"]
        enough = first and first["now"] - first["t"] >= 4
        if m["type"] == "trigger_status" and m.get("sim") and m["state"] == "armed" and enough and not fired:
            fired["t"] = time.time()
            assert _post(ui, "/api/sim/inject", {"gesture": "clench"}) == 202

    msgs = asyncio.run(_collect(ui, lambda ms: any(m["type"] == "event_done" for m in ms), 30, actions))
    ev = [m for m in msgs if m["type"] == "event"]
    done = [m for m in msgs if m["type"] == "event_done"]
    assert ev and ev[0]["detector"] == "clench" and ev[0]["source"] == "gesture"
    assert ev[0]["t"] - fired["t"] < 2.5                                   # detected promptly
    d = done[0]
    assert d["state"] == "saved" and abs(d["n_samples"] - 5 * 256) <= 2, d["n_samples"]   # -2 s / +3 s
    csv = urllib.request.urlopen(f"http://127.0.0.1:{ui}/api/events/{d['id']}/eeg.csv").read().decode()
    assert csv.startswith("t_rel_s,TP9_uv")
    listed = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{ui}/api/events").read())
    assert listed[0]["id"] == d["id"]


def test_manual_trigger_and_hotkey(running_stack):
    ui = running_stack
    sent = {}

    def actions(m):
        if m["type"] == "trigger_status" and not sent:
            sent["x"] = True
            assert _post(ui, "/api/events/trigger", {"source": "hotkey"}) == 202

    msgs = asyncio.run(_collect(ui, lambda ms: any(m["type"] == "event" for m in ms), 15, actions))
    ev = [m for m in msgs if m["type"] == "event"][0]
    assert ev["detector"] == "manual" and ev["source"] == "hotkey"


def test_guided_calibration_with_simulator(running_stack):
    ui = running_stack
    sent = {}

    def actions(m):
        if m["type"] == "trigger_status" and m.get("sim") and not sent:
            sent["x"] = True
            assert _post(ui, "/api/events/calibration", {"action": "start"}) == 202

    done = lambda ms: any(m["type"] == "calibration" and m["state"] in ("done", "failed") for m in ms)
    msgs = asyncio.run(_collect(ui, done, 40, actions))
    cal = [m for m in msgs if m["type"] == "calibration"]
    assert [m["step"] for m in cal if m["state"] == "running"].count("do") == 2
    final = cal[-1]
    assert final["state"] == "done", final
    assert final["results"]["clench"]["quality"] == "good"


@pytest.mark.skipif(not TINY_READY, reason="run `bayn models` (needs tiny)")
def test_live_transcript_and_event_transcript(running_stack):
    """Audio -> VAD -> Whisper -> browser; a manual event gets the transcript of its window attached."""
    ui = running_stack
    state = {}

    def actions(m):
        if m["type"] == "transcript" and "seg" not in state:
            state["seg"] = m                                 # speech is flowing: fire an event now
            assert _post(ui, "/api/events/trigger", {"source": "button"}) == 202
        if m["type"] == "event":
            state["event_id"] = m["id"]

    def until(ms):
        return any(m["type"] == "summary" and m["event_id"] == state.get("event_id") for m in ms)

    msgs = asyncio.run(_collect(ui, until, 120, actions))
    seg = state["seg"]
    assert seg["lang"] == "en" and "time" in seg["text"].lower()
    assert seg["t_end"] > seg["t_start"] > time.time() - 60          # unix timestamps
    assert seg["latency_s"] < 5
    status = [m for m in msgs if m["type"] == "transcript_status"][-1]
    assert status["model"] == "tiny" and status["device"] == "cpu" and status["audio_ok"]
    win = [m for m in msgs if m["type"] == "transcript_window"][-1]
    assert win["n_segments"] >= 1 and "(en)" in win["text"]
    txt = urllib.request.urlopen(f"http://127.0.0.1:{ui}/api/events/{win['event_id']}/transcript.txt").read()
    assert b"(en)" in txt
    ev = [e for e in json.loads(urllib.request.urlopen(f"http://127.0.0.1:{ui}/api/events").read())
          if e["id"] == win["event_id"]][0]
    assert ev["transcript"]["n_segments"] == win["n_segments"] and ev["state"] == "saved"
    # Phase 5: the window's transcript is summarised and streamed to the browser
    started = [m for m in msgs if m["type"] == "summary_token" and m.get("started")]
    assert started and started[0]["prompt"][1]["content"].count("[") >= 1          # prompt carries the transcript
    assert any(m["type"] == "summary_token" and m.get("text") for m in msgs)       # text appeared while writing
    summ = [m for m in msgs if m["type"] == "summary"][-1]
    assert summ["error"] is None and summ["lang"] == "en"
    assert summ["markdown"].startswith("### Event: Manual trigger") and "**Summary:**" in summ["markdown"]
    assert "not a medical assessment" in summ["markdown"]                          # code-written EEG section
    md = urllib.request.urlopen(f"http://127.0.0.1:{ui}/api/events/{win['event_id']}/summary.md").read().decode()
    assert "**Key points:**" in md


def test_bad_gesture_rejected(running_stack):
    req = urllib.request.Request(f"http://127.0.0.1:{running_stack}/api/sim/inject", method="POST",
                                 data=b'{"gesture": "wink"}', headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 400
