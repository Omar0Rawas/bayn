"""Hands-on headset check: connect, record a few seconds, report signal quality.

    python -m bayn.eeg.check                   # configured source (default: Muse via BrainFlow)
    python -m bayn.eeg.check --source lsl      # needs `muselsl stream` running
    python -m bayn.eeg.check --seconds 20 --name Muse-3EE1

Use it on any new machine (e.g. the demo laptop) before the demo. Exit code 0 = usable signal.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from bayn.common.config import load_config
from bayn.common.log import setup_logging
from bayn.eeg.filters import StreamingFilter
from bayn.eeg.quality import channel_quality, detect_mains
from bayn.eeg.sources import make_source


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Connect to the EEG source and report signal quality")
    ap.add_argument("--source", choices=["sim", "muse", "lsl"])
    ap.add_argument("--name", help="Muse name, e.g. Muse-3EE1 (BrainFlow serial_number)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--connect-timeout", type=float, default=45.0)
    args = ap.parse_args(argv)
    if args.source:
        os.environ["BAYN_EEG__SOURCE"] = args.source
    if args.name:
        os.environ["BAYN_EEG__BRAINFLOW__SERIAL_NUMBER"] = args.name
    cfg = load_config()
    log = setup_logging(cfg.logging.level, "bayn.check")

    src = make_source(cfg)
    print(f"Source: {src.name}. Connecting (up to {args.connect_timeout:.0f} s)...")
    src.start()
    try:
        t_end = time.monotonic() + args.connect_timeout
        first, last_msg = None, ""
        while first is None and time.monotonic() < t_end:
            first = src.read(timeout=1.0)
            if first is None and src.status_message and src.status_message != last_msg:
                last_msg = src.status_message
                print("  ", last_msg)
        if first is None:
            print("FAIL: no data. Headset on (lights moving)? Phone app closed? Bluetooth adapter OK?")
            return 1
        print(f"Receiving data. Recording {args.seconds:.0f} s - sit still, relax your jaw, eyes open...")
        chunks = [first]
        t_rec = time.monotonic() + args.seconds
        while time.monotonic() < t_rec:
            if (c := src.read(timeout=1.0)) is not None:
                chunks.append(c)
    finally:
        src.stop()

    raw = np.hstack([c.data for c in chunks])
    fs = chunks[0].fs
    span = chunks[-1].t_end - chunks[0].t0
    filt = StreamingFilter(fs, raw.shape[0], tuple(cfg.eeg.bandpass_hz), cfg.eeg.notch_hz).process(raw)
    filt = filt[:, int(fs):] if filt.shape[1] > 2 * fs else filt   # drop filter warm-up
    print(f"\nSamples: {raw.shape[1]} over {span:.1f} s -> {raw.shape[1] / max(span, 1e-9):.1f} Hz "
          f"(nominal {fs:.0f} Hz), chunks: {len(chunks)}")
    mains = detect_mains(raw - raw.mean(axis=1, keepdims=True), fs)
    print(f"Mains hum: {f'{mains} Hz' if mains else 'not clearly present (fine)'}")
    print("Channel quality (1-40 Hz, robust amplitude):")
    quality = channel_quality(filt, src.channels)
    for q in quality:
        print(f"  {q.channel:5s} {q.level:5s} {q.noise_uv:7.1f} µV  {q.reason}")
    bad = [q.channel for q in quality if q.level == "bad"]
    if bad:
        print(f"\nAdjust the headband / wet the sensors for: {', '.join(bad)}")
    log.debug("done")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
