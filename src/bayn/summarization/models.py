"""Pick and download the summary model for this machine (via the local Ollama server).

    python -m bayn.summarization.models          # pull the model 'auto' picks here
    python -m bayn.summarization.models --all    # pull both (e.g. before moving to a laptop)

auto: qwen2.5:7b with an NVIDIA GPU of >= 10 GB (or Apple Silicon with >= 16 GB), else qwen2.5:3b.
Measured on the classroom scenarios (docs/TESTING.md, Phase 5): 7b covers 82 % of key facts with
structured output; 3b only 50 %, but it is the only practical choice on a CPU-only laptop.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import httpx

from bayn.common.config import SummaryConfig, load_config
from bayn.summarization.ollama_client import OllamaClient, ollama_env, ollama_serve_command

LARGE, SMALL = "qwen2.5:7b", "qwen2.5:3b"


def gpu_memory_mb() -> int:
    """Total memory of the largest NVIDIA GPU, 0 if none."""
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return max(int(x) for x in out.split()) if out.strip() else 0
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def apple_silicon_memory_gb() -> float:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return 0.0
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout
        return int(out) / 2**30
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


# Whisper large-v3-turbo (~2.5 GB) and the LLM share the GPU: qwen2.5:7b (~4.8 GB) needs ~10 GB total
# to run without spilling; on an 8 GB laptop GPU the 3b model (~2.2 GB) is the safe choice.
GPU_MB_FOR_LARGE = 10_000


def auto_model() -> str:
    return LARGE if gpu_memory_mb() >= GPU_MB_FOR_LARGE or apple_silicon_memory_gb() >= 16 else SMALL


def resolve_model(cfg: SummaryConfig, available: list[str] | None = None) -> str:
    """Configured model, or the auto choice; falls back to whichever Qwen model is pulled."""
    want = auto_model() if cfg.model == "auto" else cfg.model
    if available is not None and cfg.model == "auto" and want not in available:
        for alt in (LARGE, SMALL):
            if alt in available:
                return alt
    return want


def pull(host: str, model: str) -> None:
    """Download a model through the Ollama REST API, printing progress."""
    last = ""
    with httpx.stream("POST", f"{host.rstrip('/')}/api/pull", json={"model": model}, timeout=None) as r:
        for line in r.iter_lines():
            if not line:
                continue
            m = json.loads(line)
            if "error" in m:
                raise RuntimeError(m["error"])
            status = m.get("status", "")
            if m.get("total"):
                status += f" {100 * m.get('completed', 0) / m['total']:.0f}%"
            if status != last:
                print(f"\r  {model}: {status:<60}", end="", flush=True)
                last = status
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download the summary model(s) with Ollama")
    ap.add_argument("--all", action="store_true", help=f"pull {LARGE} and {SMALL}")
    args = ap.parse_args(argv)
    cfg = load_config().summary
    server = None
    if cmd := ollama_serve_command(cfg.host):                  # not running: start it just for the download
        server = subprocess.Popen(cmd, env={**os.environ, **ollama_env()},
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            if OllamaClient(cfg.host).version():
                break
            time.sleep(0.5)
    try:
        if OllamaClient(cfg.host).version() is None:
            print(f"Ollama is not available at {cfg.host}. Install it from https://ollama.com, then retry.")
            return 1
        names = [LARGE, SMALL] if args.all else [resolve_model(cfg)]
        print(f"Summary model(s) for this machine: {', '.join(names)}"
              f"  -> {ollama_env().get('OLLAMA_MODELS', 'Ollama default folder')}")
        for n in names:
            pull(cfg.host, n)
    finally:
        if server:
            server.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
