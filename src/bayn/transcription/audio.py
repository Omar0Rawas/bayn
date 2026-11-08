"""Audio capture process: microphone (or WAV file) -> 16 kHz chunks on the bus.

Runs natively on the host (the microphone is not reachable from containers/VMs).

    python -m bayn.transcription.audio [--source mic|file] [--file path.wav] [--list-devices]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

from bayn.common.bus import Publisher
from bayn.common.config import ROOT, AppConfig, load_config
from bayn.common.log import setup_logging
from bayn.common.schemas import TOPIC_AUDIO_LEVEL, TOPIC_AUDIO_PCM, TOPIC_AUDIO_STATUS
from bayn.transcription.audio_sources import AudioSource, FileSource, MicSource

log = logging.getLogger("bayn.audio")
LEVEL_EVERY_S = 0.1
STATUS_EVERY_S = 1.0
RETRY_S = 3.0


def make_audio_source(cfg: AppConfig) -> AudioSource:
    a = cfg.audio
    if a.source == "file":
        path = a.file if os.path.isabs(a.file) else ROOT / a.file
        return FileSource(path, loop=a.loop, chunk_ms=a.chunk_ms)
    return MicSource(a.device, a.chunk_ms)


def level_db(x: np.ndarray) -> tuple[float, float]:
    """(RMS dBFS, peak dBFS), floored at -90."""
    if x.size == 0:
        return -90.0, -90.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(x)))
    return max(-90.0, 20 * np.log10(rms + 1e-12)), max(-90.0, 20 * np.log10(peak + 1e-12))


def run(cfg: AppConfig) -> None:
    pub = Publisher(cfg.bus.bind_host, cfg.bus.audio_pub_port)
    src: AudioSource | None = None
    last_level = last_status = 0.0
    last_audio = time.time()
    lvl_buf: list[np.ndarray] = []
    error = ""
    try:
        while True:
            if src is None:                                   # (re)open the input
                try:
                    src = make_audio_source(cfg)
                    src.start()
                    error = ""
                    last_audio = time.time()
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
                    log.warning("audio input unavailable (%s) - retrying in %.0f s", error, RETRY_S)
                    pub.send(TOPIC_AUDIO_STATUS, {"ok": False, "source": cfg.audio.source, "device": "",
                                                  "message": error, "t": time.time()})
                    src = None
                    time.sleep(RETRY_S)
                    continue
            chunk = src.read(timeout=1.0)
            now = time.time()
            if chunk is not None:
                pub.send(TOPIC_AUDIO_PCM, chunk.to_msg())
                lvl_buf.append(chunk.samples)
                last_audio = now
            elif src.name == "mic" and now - last_audio > 5:                       # mic unplugged?
                log.warning("no audio for 5 s - reopening the microphone")
                src.stop()
                src = None
                last_audio = now
                continue
            if now - last_level >= LEVEL_EVERY_S and lvl_buf:
                last_level = now
                rms, peak = level_db(np.concatenate(lvl_buf))
                lvl_buf.clear()
                pub.send(TOPIC_AUDIO_LEVEL, {"t": now, "rms_db": round(rms, 1), "peak_db": round(peak, 1)})
            if now - last_status >= STATUS_EVERY_S:
                last_status = now
                ok = chunk is not None
                pub.send(TOPIC_AUDIO_STATUS, {"ok": ok, "source": src.name, "device": src.description,
                                              "message": "" if ok else "no audio arriving", "t": now})
    finally:
        if src:
            src.stop()
        pub.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bayn audio capture")
    ap.add_argument("--source", choices=["mic", "file"])
    ap.add_argument("--file", help="WAV file for --source file")
    ap.add_argument("--device", help="microphone name (part of it) or index")
    ap.add_argument("--list-devices", action="store_true", help="print input devices and exit")
    args = ap.parse_args(argv)
    if args.list_devices:
        for d in MicSource.list_devices():
            print(f"[{d['index']:2d}] {d['name']}  ({d['channels']} ch, {d['default_sr']:.0f} Hz)")
        return 0
    if args.source:
        os.environ["BAYN_AUDIO__SOURCE"] = args.source
    if args.file:
        os.environ["BAYN_AUDIO__FILE"] = args.file
    if args.device:
        os.environ["BAYN_AUDIO__DEVICE"] = args.device
    cfg = load_config()
    setup_logging(cfg.logging.level)
    try:
        run(cfg)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
