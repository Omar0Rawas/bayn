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


@pytest.fixture
def running_stack():
    ui, pub, cmd = _free_port(), _free_port(), _free_port()
    env = dict(os.environ, BAYN_EEG__SOURCE="sim", BAYN_UI__PORT=str(ui),
               BAYN_BUS__EEG_PUB_PORT=str(pub), BAYN_BUS__EEG_CMD_PORT=str(cmd),
               BAYN_SIMULATOR__SEED="5", BAYN_SIMULATOR__MAINS_HZ="60",
               BAYN_SIMULATOR__AUTO_GESTURES__ENABLED="false", BAYN_LOGGING__LEVEL="WARNING")
    procs = [subprocess.Popen([sys.executable, "-m", m], env=env)
             for m in ("bayn.eeg.bridge", "bayn.ui.server")]
    try:
        _wait_http(f"http://127.0.0.1:{ui}/api/health")
        yield ui
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=10)


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


def test_bad_gesture_rejected(running_stack):
    req = urllib.request.Request(f"http://127.0.0.1:{running_stack}/api/sim/inject", method="POST",
                                 data=b'{"gesture": "wink"}', headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 400
