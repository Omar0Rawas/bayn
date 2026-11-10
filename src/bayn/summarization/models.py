"""Pick and download the summary model for this machine (via the local Ollama server).

    python -m bayn.summarization.models          # pull the model 'auto' picks here
    python -m bayn.summarization.models --all    # pull both (e.g. before moving to a laptop)

auto (measured on the recorded corpus, docs/BENCHMARK.md and docs/BENCHMARK_LAPTOP.md):
  gemma4:e2b  with an NVIDIA GPU of >= 6 GB, Apple Silicon, no NVIDIA GPU and >= 7 GB RAM (8 GB laptops),
              or a smaller NVIDIA GPU and >= 12 GB RAM: best measured quality (77-83 % of facts on real
              transcripts on the dev PC; 70 % via Whisper small on the demo laptop's CPU), 3.0 GB on the GPU;
              on the demo laptop (i7-1355U, 8 GB) ~1.7 GB resident RAM, ~19 s per summary.
  qwen3:4b    otherwise (< 7 GB RAM, or a small NVIDIA GPU with < 12 GB RAM): not measured there.
              On the demo laptop it was worse (45 % facts via small, 35 s per summary, 19/96 empty summaries).
Previous defaults: qwen2.5:7b (68 %, 4.8 GB) and qwen2.5:3b (44-52 %).
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys

import httpx

from bayn.common.config import SummaryConfig, load_config
from bayn.summarization.ollama_client import local_server, ollama_env

LARGE, SMALL = "gemma4:e2b", "qwen3:4b"
FALLBACKS = (LARGE, SMALL, "qwen2.5:7b", "qwen2.5:3b")      # if auto's choice isn't pulled, use one that is


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


def system_ram_gb() -> float:
    """Total physical memory (GB), 0 if unknown."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                           [(n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                                              "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                                                              "ullAvailExtendedVirtual")]
            m = MemStatus()
            m.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 2**30
        import os

        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (OSError, ValueError, AttributeError):
        return 0.0


# Whisper (turbo ~2.3 GB) and the LLM share the GPU: gemma4:e2b (~3.0 GB measured) fits a 6 GB GPU with
# Whisper small, an 8 GB GPU with turbo.
GPU_MB_FOR_LARGE = 6_000
# No NVIDIA GPU (demo laptop, 8 GB RAM, docs/BENCHMARK_LAPTOP.md): gemma4:e2b keeps only ~1.7 GB resident (its
# per-layer embedding tables are memory-mapped; the dev PC showed 4.7 GB only because it had RAM to spare), is
# ~2x faster than qwen3:4b (19 vs 35 s per summary) and finds more facts on every transcript (small: 70 vs 45 %).
# An "8 GB" machine reports ~7.6 GB, hence 7. Below that: not measured, keep qwen3:4b.
RAM_GB_FOR_LARGE_CPU_ONLY = 7
# With a small NVIDIA GPU (< 6 GB) Ollama would split the model across a GPU that Whisper already uses: not
# measured, so the old, cautious threshold stays.
RAM_GB_FOR_LARGE = 12


def auto_model() -> str:
    gpu = gpu_memory_mb()
    if gpu >= GPU_MB_FOR_LARGE or apple_silicon_memory_gb() >= 8:
        return LARGE
    if system_ram_gb() >= (RAM_GB_FOR_LARGE if gpu else RAM_GB_FOR_LARGE_CPU_ONLY):
        return LARGE
    return SMALL


def resolve_model(cfg: SummaryConfig, available: list[str] | None = None) -> str:
    """Configured model, or the auto choice; falls back to whichever Qwen model is pulled."""
    want = auto_model() if cfg.model == "auto" else cfg.model
    if available is not None and cfg.model == "auto" and want not in available:
        for alt in FALLBACKS:
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
    ap.add_argument("names", nargs="*", help="specific Ollama models to pull (e.g. for `bayn benchmark`)")
    args = ap.parse_args(argv)
    cfg = load_config().summary
    with local_server(cfg.host) as ok:               # not running: started just for the download
        if not ok:
            print(f"Ollama is not available at {cfg.host}. Install it from https://ollama.com, then retry.")
            return 1
        names = args.names or ([LARGE, SMALL] if args.all else [resolve_model(cfg)])
        print(f"Summary model(s): {', '.join(names)}"
              f"  -> {ollama_env().get('OLLAMA_MODELS', 'Ollama default folder')}")
        failed = []
        for n in names:
            try:
                pull(cfg.host, n)
            except RuntimeError as e:
                print(f"\n  {n}: {e}")
                failed.append(n)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
