"""Record the microphone to a WAV file (for evaluating transcription on your own voice).

    python -m bayn.transcription.record mx-pi            # a corpus script: opens it in big type, Enter stops
    python -m bayn.transcription.record my_test --seconds 45
    -> data/recordings/<name>.wav   (gitignored: your voice stays on this machine)
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np

from bayn import corpus
from bayn.common.config import DATA_DIR, load_config
from bayn.transcription.audio import level_db
from bayn.transcription.audio_sources import MicSource, write_wav


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Record the microphone to data/recordings/<name>.wav")
    ap.add_argument("name", help="a corpus id (e.g. mx-pi, see `python -m bayn.corpus list`) or any file name")
    ap.add_argument("--seconds", type=float, default=120.0, help="maximum length (Enter stops earlier)")
    ap.add_argument("--device", help="microphone (part of its name, or index); default = config / system default")
    ap.add_argument("--no-browser", action="store_true", help="don't open the script's reading page")
    args = ap.parse_args(argv)
    cfg = load_config()
    out = DATA_DIR / "recordings" / f"{args.name}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)

    script = corpus.get(args.name)
    if script:
        print("=" * 70)
        print(f"{args.name}: {corpus.LANG_NAMES[script['language']]} - {corpus.topic(script)['title']}"
              f"  (~{corpus.est_seconds(script)} s)\n")
        corpus.print_script(script)
        print("=" * 70)
        if not args.no_browser:
            page = corpus.open_reading_page(script)
            print(f"The script is open in your browser ({page.name}). Put it next to this window.")
    if out.exists():
        print(f"(this replaces the existing {out.name})")
    if script:
        try:
            input("Click back into this window and press Enter to start (Ctrl+C cancels)... ")
        except (KeyboardInterrupt, EOFError):
            print("\ncancelled - nothing saved")
            return 1

    mic = MicSource(args.device if args.device is not None else cfg.audio.device)
    mic.start()
    print(f"Microphone: {mic.description}")
    for n in (3, 2, 1):
        print(f"  starting in {n}...", flush=True)
        time.sleep(1)
    print("RECORDING - read now. Press Enter when you're done (Ctrl+C cancels).")
    stop = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.readline(), stop.set()), daemon=True).start()
    chunks, t_end = [], time.time() + args.seconds
    try:
        while time.time() < t_end and not stop.is_set():
            c = mic.read(timeout=1.0)
            if c is None:
                continue
            chunks.append(c.samples)
            rms, _ = level_db(c.samples)
            bar = "#" * max(0, int((rms + 60) / 2))
            print(f"\r  {sum(len(x) for x in chunks) / 16000:5.1f} s  level {rms:6.1f} dBFS |{bar:<30}|",
                  end="", flush=True)
    except KeyboardInterrupt:
        mic.stop()
        print("\ncancelled - nothing saved")
        return 1
    mic.stop()
    audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    write_wav(out, audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    print(f"\nSaved {len(audio) / 16000:.1f} s to {out}")
    if peak < 0.02:
        print("WARNING: very quiet recording - check the microphone (`bayn mics`) and record again.")
    elif peak > 0.99:
        print("WARNING: the recording clips (too loud) - move back a little or lower the mic gain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
