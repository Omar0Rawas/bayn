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


def _check_speech(cfg) -> None:
    """Whisper packages + downloaded models + CUDA usability + microphone."""
    try:
        importlib.import_module("faster_whisper")
        from bayn.transcription.engine import CPU_MODEL, GPU_MODEL, add_cuda_dll_dirs, cuda_available, model_dir
    except ImportError:
        _add(WARN, "package faster_whisper", "missing - transcription disabled (run setup)")
        return
    add_cuda_dll_dirs()
    gpu = cuda_available()
    wanted = cfg.transcription.model if cfg.transcription.model != "auto" else (GPU_MODEL if gpu else CPU_MODEL)
    have = [n for n in (GPU_MODEL, CPU_MODEL, "medium", "base", "tiny") if (model_dir(n) / "model.bin").exists()]
    _add(OK if wanted in have else FAIL, "Whisper model",
         f"{wanted} on {'GPU' if gpu else 'CPU'} - " + ("downloaded" if wanted in have else "missing: run `bayn models`")
         + (f" (also here: {', '.join(n for n in have if n != wanted)})" if len(have) > 1 else ""))
    if gpu and CPU_MODEL not in have:
        _add(WARN, "Whisper CPU fallback", f"{CPU_MODEL} not downloaded - no fallback if the GPU fails")
    if gpu:
        try:
            importlib.import_module("nvidia.cudnn")
            _add(OK, "CUDA libraries", "cuDNN/cuBLAS installed")
        except ImportError:
            _add(WARN, "CUDA libraries", "nvidia-cudnn-cu12 missing - Whisper will fall back to CPU (pip install -r requirements/gpu.txt)")
    if cfg.audio.source == "mic":
        try:
            import sounddevice as sd

            dev = sd.query_devices(kind="input")
            n = len([d for d in sd.query_devices() if d["max_input_channels"] > 0])
            _add(OK, "Microphone", f"default: {dev['name']} ({n} input devices; `bayn mics` lists them)")
        except Exception as e:
            _add(WARN, "Microphone", f"no input device ({e}) - plug one in, or use --audio file")


def _check_ollama(cfg) -> None:
    """Local LLM runtime reachable + the summary model this machine should use is pulled."""
    from bayn.summarization.models import resolve_model
    from bayn.summarization.ollama_client import OllamaClient

    from bayn.summarization.ollama_client import find_ollama

    client = OllamaClient(cfg.summary.host)
    version = client.version()
    if version is None:
        if find_ollama():
            _add(INFO, "Ollama", "installed, not running - `bayn run`/`demo` start it automatically "
                                 "(model check skipped)")
        else:
            _add(WARN, "Ollama", "not installed (https://ollama.com, or `bayn setup`) - summaries are off until then")
        return
    if not cfg.summary.host.startswith(("http://127.0.0.1", "http://localhost")):
        _add(WARN, "Ollama host", f"{cfg.summary.host} is not local - Bayn is meant to run offline")
    available = client.models()
    model = resolve_model(cfg.summary, available)
    have = model in available or f"{model}:latest" in available
    _add(OK if have else FAIL, "Summary model", f"{model} via Ollama {version} - "
         + ("ready" if have else "missing: run `bayn models`"))


def _folder_gb(path) -> float:
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30
    except OSError:
        return 0.0


def _check_disk() -> None:
    """Where the big files live and how much room is left (laptops often have a small C: drive)."""
    from pathlib import Path

    from bayn.common.config import DATA_DIR, MODELS_DIR

    from bayn.summarization.ollama_client import ollama_env

    ollama_dir = Path(ollama_env().get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models")
    places = {"Whisper models": MODELS_DIR / "whisper", "Ollama models": ollama_dir, "Bayn data": DATA_DIR}
    for what, path in places.items():
        if not path.exists():
            continue
        free = shutil.disk_usage(path).free / 2**30
        level = OK if free >= 5 else WARN
        _add(level, what, f"{_folder_gb(path):.1f} GB in {path} - {free:.0f} GB free on that drive"
             + ("" if free >= 5 else " (low! set OLLAMA_MODELS / BAYN_MODELS_DIR to a bigger drive)"))


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
    _check_speech(cfg)

    # Later phases - reported now so a new machine can be prepared early.
    _check_ollama(cfg)
    _check_disk()

    width = max(len(w) for _, w, _ in _results)
    for level, what, detail in _results:
        print(f"  [{level:>4}] {what:<{width}}  {detail}")
    failed = any(l == FAIL for l, _, _ in _results)
    print("\nResult:", "problems found - see FAIL lines" if failed else "ready")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(check())
