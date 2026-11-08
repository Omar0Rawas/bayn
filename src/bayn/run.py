"""One-command launcher: starts every native Bayn process and the dashboard.

    python -m bayn.run --demo          # simulated EEG + scripted gestures (no hardware)
    python -m bayn.run --source muse   # real headset (Phase 2)

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
    ap.add_argument("--no-browser", action="store_true", help="do not open the dashboard")
    args = ap.parse_args(argv)

    env = dict(os.environ)
    if args.demo:
        env["BAYN_EEG__SOURCE"] = "sim"
        env["BAYN_SIMULATOR__AUTO_GESTURES__ENABLED"] = "true"
    if args.source:
        env["BAYN_EEG__SOURCE"] = args.source
    os.environ.update(env)
    cfg = load_config()

    procs: dict[str, subprocess.Popen] = {}
    py = sys.executable
    for name, module in [("eeg-bridge", "bayn.eeg.bridge"), ("dashboard", "bayn.ui.server")]:
        log.info("starting %s", name)
        procs[name] = subprocess.Popen([py, "-m", module], env=env)

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
            for name, p in procs.items():
                if p.poll() is not None:
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
