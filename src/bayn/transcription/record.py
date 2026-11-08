"""Record the microphone to a WAV file (for evaluating transcription on your own voice).

    python -m bayn.transcription.record classroom_mixed --seconds 45
    -> data/recordings/classroom_mixed.wav   (gitignored: your voice stays on this machine)
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from bayn.common.config import DATA_DIR, load_config
from bayn.transcription.audio import level_db
from bayn.transcription.audio_sources import MicSource, write_wav


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Record the microphone to data/recordings/<name>.wav")
    ap.add_argument("name", help="file name without .wav, e.g. classroom_mixed")
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--device", help="microphone (part of its name, or index); default = config / system default")
    args = ap.parse_args(argv)
    cfg = load_config()
    out = DATA_DIR / "recordings" / f"{args.name}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)

    mic = MicSource(args.device if args.device is not None else cfg.audio.device)
    mic.start()
    print(f"Microphone: {mic.description}")
    for n in (3, 2, 1):
        print(f"  starting in {n}...", flush=True)
        time.sleep(1)
    print(f"RECORDING {args.seconds:.0f} s - read your script now. (Ctrl+C stops early)")
    chunks, end = [], time.time() + args.seconds
    try:
        while time.time() < end:
            c = mic.read(timeout=1.0)
            if c is None:
                continue
            chunks.append(c.samples)
            rms, _ = level_db(c.samples)
            bar = "#" * max(0, int((rms + 60) / 2))
            print(f"\r  {end - time.time():4.0f} s left  level {rms:6.1f} dBFS |{bar:<30}|", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
    audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    write_wav(out, audio)
    print(f"\nSaved {len(audio) / 16000:.1f} s to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
