"""Pre-flight check for a machine: run before a demo on any PC/laptop.

    python -m bayn.doctor

Reports, never changes anything. Exit code 1 only for problems that block the demo.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import socket
import subprocess
import sys

from bayn.common.config import load_config

OK, WARN, FAIL, INFO = "ok", "warn", "FAIL", "info"
_results: list[tuple[str, str, str]] = []


def _add(level: str, what: str, detail: str = "") -> None:
    _results.append((level, what, detail))


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _cmd(*args: str) -> str | None:
    if not shutil.which(args[0]):
        return None
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _check_bluetooth(seconds: float = 5.0) -> None:
    """Short BLE scan: is there a working adapter, and is a Muse advertising?"""
    try:
        import asyncio

        from bleak import BleakScanner
        from bleak.exc import BleakError
    except ImportError:
        _add(INFO, "Bluetooth", "bleak not installed - skipped scan (use `check` to test the headset)")
        return
    try:
        found = asyncio.run(BleakScanner.discover(timeout=seconds, return_adv=True))
    except (BleakError, OSError) as e:
        _add(WARN, "Bluetooth", f"no usable adapter ({e}) - plug in / enable Bluetooth; on Windows 10 "
                                "a USB dongle may need its vendor driver")
        return
    muses = sorted(((adv.rssi, d.name or adv.local_name) for d, adv in found.values()
                    if "muse" in (d.name or adv.local_name or "").lower()), reverse=True)
    _add(OK, "Bluetooth adapter", f"working, {len(found)} BLE devices visible")
    if muses:
        _add(OK, "Muse headset", ", ".join(f"{n} ({r} dBm)" for r, n in muses)
             + (" - several found: set eeg.brainflow.serial_number" if len(muses) > 1 else ""))
    else:
        _add(WARN, "Muse headset", "not seen - turn it on (lights moving), close the phone app, move closer")


def check() -> int:
    v = sys.version_info
    _add(OK if v >= (3, 12) else FAIL, "Python", f"{platform.python_version()} ({'needs 3.12+' if v < (3, 12) else 'supported'})")
    _add(INFO, "OS", f"{platform.system()} {platform.release()} ({platform.machine()})")

    for mod in ("numpy", "scipy", "zmq", "msgpack", "yaml", "pydantic", "fastapi", "uvicorn"):
        try:
            m = importlib.import_module(mod)
            _add(OK, f"package {mod}", getattr(m, "__version__", ""))
        except ImportError:
            _add(FAIL, f"package {mod}", "missing - run setup")

    cfg = load_config()
    for name, host, port in [("dashboard port", cfg.ui.host, cfg.ui.port),
                             ("EEG data port", cfg.bus.bind_host, cfg.bus.eeg_pub_port),
                             ("EEG command port", cfg.bus.bind_host, cfg.bus.eeg_cmd_port)]:
        free = _port_free(host, port)
        _add(OK if free else WARN, f"{name} {port}", "free" if free else "in use - is Bayn already running? change it in config/local.yaml")

    gpu = _cmd("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader")
    _add(INFO, "GPU", f"{gpu} - Whisper will use CUDA" if gpu else "none detected - Whisper will run on CPU (fine with the 'small' model)")

    _add(INFO, "EEG source (config)", cfg.eeg.source + (" - no headset needed" if cfg.eeg.source == "sim" else ""))
    for mod, why in (("brainflow", "Muse via BrainFlow"), ("pylsl", "LSL input"), ("muselsl", "muse-lsl fallback")):
        try:
            importlib.import_module(mod)
            _add(OK, f"package {mod}", why)
        except ImportError:
            _add(FAIL if mod == "brainflow" else WARN, f"package {mod}", f"missing ({why}) - run setup")
    if cfg.eeg.source != "sim":
        _check_bluetooth()

    # Later phases - reported now so a new machine can be prepared early.
    _add(INFO if shutil.which("ollama") else WARN, "Ollama", "installed" if shutil.which("ollama") else "not installed (needed from Phase 5: https://ollama.com)")
    _add(INFO if shutil.which("docker") else WARN, "Docker", "installed" if shutil.which("docker") else "not installed (optional, Phase 7)")

    width = max(len(w) for _, w, _ in _results)
    for level, what, detail in _results:
        print(f"  [{level:>4}] {what:<{width}}  {detail}")
    failed = any(l == FAIL for l, _, _ in _results)
    print("\nResult:", "problems found - see FAIL lines" if failed else "ready")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(check())
