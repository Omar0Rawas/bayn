"""One-command launcher: starts every native Bayn process and the dashboard.

    python -m bayn.run                 # real Muse 2 via BrainFlow (default source)
    python -m bayn.run --source lsl    # Muse via muse-lsl (started automatically)
    python -m bayn.run --demo          # simulated EEG + scripted gestures (no hardware)

Each component is a separate process (same as under Docker), so a crash in one
does not take down the others. Ctrl+C stops everything.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser

from bayn.common.config import load_config
from bayn.common.log import setup_logging
from bayn.summarization.ollama_client import ollama_env, ollama_serve_command

log = setup_logging(name="bayn.run")


def _wait_http(url: str, timeout_s: float = 20.0) -> bool:
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the Bayn pipeline")
    ap.add_argument("--demo", action="store_true", help="simulated EEG with scripted gestures")
    ap.add_argument("--source", choices=["sim", "muse", "lsl"], help="EEG source (default from config)")
    ap.add_argument("--audio", choices=["mic", "file"], help="mic (default) or file (samples/audio/demo.wav)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the dashboard")
    args = ap.parse_args(argv)

    env = dict(os.environ)
    if args.demo:
        env["BAYN_EEG__SOURCE"] = "sim"
        env["BAYN_SIMULATOR__AUTO_GESTURES__ENABLED"] = "true"
        env.setdefault("BAYN_EVENTS__GESTURE", "both")     # scripted gestures alternate clench / double blink
    if args.source:
        env["BAYN_EEG__SOURCE"] = args.source
    if args.audio:
        env["BAYN_AUDIO__SOURCE"] = args.audio
    os.environ.update(env)
    cfg = load_config()

    py = sys.executable
    # name -> (command, restart_on_exit). Core processes stop the whole run if they die;
    # the muse-lsl helper exits whenever the headset drops, so it is simply restarted.
    specs: dict[str, tuple[list[str], bool]] = {
        "eeg-bridge": ([py, "-m", "bayn.eeg.bridge"], False),
        "events": ([py, "-m", "bayn.events.service"], False),
        "dashboard": ([py, "-m", "bayn.ui.server"], False),
        # speech side: restarted if it ever dies, so it can't take EEG/triggering down with it
        "audio": ([py, "-m", "bayn.transcription.audio"], True),
        "transcriber": ([py, "-m", "bayn.transcription.service"], True),
        "summarizer": ([py, "-m", "bayn.summarization.service"], True),
    }
    if cfg.eeg.source == "lsl" and cfg.eeg.lsl.autostart_muselsl:
        cmd = [py, "-m", "muselsl", "stream"]
        bf = cfg.eeg.brainflow
        if bf.mac_address:
            cmd += ["--address", bf.mac_address]
        elif bf.serial_number:
            cmd += ["--name", bf.serial_number]
        specs["muselsl"] = (cmd, True)
    # Local LLM: use a running Ollama if there is one, otherwise run `ollama serve` ourselves
    # (doesn't depend on the Ollama tray app, which may not be running on the demo laptop).
    if ollama_cmd := ollama_serve_command(cfg.summary.host):
        specs["ollama"] = (ollama_cmd, True)
        env.update(ollama_env())                               # local only + the user's model folder
        log.info("Ollama not running - starting it (models in %s)", env.get("OLLAMA_MODELS", "Ollama's default folder"))
    log.info("EEG source: %s", cfg.eeg.source + (" (simulated, scripted gestures)" if args.demo else ""))

    procs: dict[str, subprocess.Popen] = {}
    for name, (cmd, _) in specs.items():
        log.info("starting %s", name)
        procs[name] = subprocess.Popen(cmd, env=env)

    url = f"http://{'127.0.0.1' if cfg.ui.host in ('0.0.0.0', '') else cfg.ui.host}:{cfg.ui.port}"
    if _wait_http(url + "/api/health"):
        log.info("dashboard ready: %s  (Ctrl+C to stop)", url)
        if cfg.ui.open_browser and not args.no_browser:
            webbrowser.open(url)
    else:
        log.error("dashboard did not come up on %s", url)

    code = 0
    try:
        while True:
            for name, p in list(procs.items()):
                if p.poll() is None:
                    continue
                cmd, restart = specs[name]
                if restart:
                    log.warning("%s exited (code %s) - restarting in 3 s", name, p.returncode)
                    time.sleep(3)
                    procs[name] = subprocess.Popen(cmd, env=env)
                    continue
                log.error("%s exited with code %s - shutting down", name, p.returncode)
                code = p.returncode or 1
                raise KeyboardInterrupt
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("stopping...")
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
    return code


if __name__ == "__main__":
    sys.exit(main())
