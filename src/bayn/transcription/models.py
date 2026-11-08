"""Download Whisper models once (setup step). Everything else then runs offline.

    python -m bayn.transcription.models            # what this machine needs (+ tiny for tests)
    python -m bayn.transcription.models --all      # GPU and CPU models (e.g. to copy to a laptop)
    python -m bayn.transcription.models small      # a specific model
"""

from __future__ import annotations

import argparse
import shutil
import sys

from bayn.common.config import MODELS_DIR
from bayn.transcription.engine import CPU_MODEL, GPU_MODEL, add_cuda_dll_dirs, cuda_available, model_dir

TEST_MODEL = "tiny"


def download(name: str) -> None:
    from faster_whisper import download_model

    dest = model_dir(name)
    if (dest / "model.bin").exists():
        print(f"  {name}: already present ({dest})")
        return
    print(f"  {name}: downloading to {dest} ...", flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        download_model(name, output_dir=str(dest))
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)   # never leave a half-downloaded model behind
        raise
    print(f"  {name}: done")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download Whisper models for offline use")
    ap.add_argument("names", nargs="*", help="model names (default: what this machine needs)")
    ap.add_argument("--all", action="store_true", help=f"{GPU_MODEL} + {CPU_MODEL} + {TEST_MODEL}")
    args = ap.parse_args(argv)
    add_cuda_dll_dirs()
    if args.names:
        names = args.names
    elif args.all:
        names = [GPU_MODEL, CPU_MODEL, TEST_MODEL]
    else:
        # CPU model always: it is the fallback if the GPU can't be used at demo time
        names = ([GPU_MODEL] if cuda_available() else []) + [CPU_MODEL, TEST_MODEL]
    print(f"Whisper models -> {MODELS_DIR / 'whisper'}")
    for n in names:
        download(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
