"""Audio inputs: live microphone (default) or a WAV file played in real time.

Both deliver mono float32 AudioChunks at 16 kHz with unix timestamps.
"""

from __future__ import annotations

import logging
import queue
import time
import wave
from abc import ABC, abstractmethod
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from bayn.common.schemas import AUDIO_SR, AudioChunk

log = logging.getLogger("bayn.audio")


def to_mono_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """(n,) or (n, ch) float -> mono float32 at 16 kHz."""
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != AUDIO_SR:
        g = gcd(AUDIO_SR, sr)
        x = resample_poly(x, AUDIO_SR // g, sr // g)
    return x.astype(np.float32)


class AudioSource(ABC):
    name = "base"

    def __init__(self) -> None:
        self.description = ""
        self.error = ""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def read(self, timeout: float = 1.0) -> AudioChunk | None: ...

    @abstractmethod
    def stop(self) -> None: ...


class MicSource(AudioSource):
    """Live microphone via sounddevice (PortAudio). Device: None = default, name substring or index."""

    name = "mic"

    def __init__(self, device: str | int | None = None, chunk_ms: int = 100) -> None:
        super().__init__()
        self.device = device
        self.chunk_ms = chunk_ms
        self._q: queue.Queue[AudioChunk] = queue.Queue(maxsize=600)
        self._stream = None
        self._seq = 0
        self._sr = AUDIO_SR

    @staticmethod
    def list_devices() -> list[dict]:
        import sounddevice as sd

        return [{"index": i, "name": d["name"], "channels": d["max_input_channels"],
                 "default_sr": d["default_samplerate"]}
                for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]

    def _resolve_device(self):
        import sounddevice as sd

        if self.device is None or self.device == "":
            return None
        if isinstance(self.device, int) or str(self.device).isdigit():
            return int(self.device)
        matches = [d for d in self.list_devices() if str(self.device).lower() in d["name"].lower()]
        if not matches:
            raise ValueError(f"no input device matching {self.device!r}")
        return matches[0]["index"]

    def _callback(self, indata, frames, t_info, status) -> None:
        now = time.time()
        if status:
            log.debug("audio status: %s", status)
        x = to_mono_16k(indata.copy(), self._sr)
        chunk = AudioChunk(now - frames / self._sr, x, AUDIO_SR, self._seq)
        self._seq += 1
        if self._q.full():
            self._q.get_nowait()
        self._q.put_nowait(chunk)

    def start(self) -> None:
        import sounddevice as sd

        dev = self._resolve_device()
        info = sd.query_devices(dev, "input")
        # ask for 16 kHz directly; fall back to the device's own rate and resample
        for sr in (AUDIO_SR, int(info["default_samplerate"])):
            try:
                self._sr = sr
                self._stream = sd.InputStream(device=dev, channels=1, samplerate=sr, dtype="float32",
                                              blocksize=int(sr * self.chunk_ms / 1000), callback=self._callback)
                self._stream.start()
                break
            except Exception as e:  # PortAudio rejects the rate on some host APIs
                log.debug("opening %s at %d Hz failed: %s", info["name"], sr, e)
                self._stream = None
        if self._stream is None:
            raise RuntimeError(f"could not open microphone {info['name']!r}")
        self.description = f"{info['name']} @ {self._sr} Hz"
        log.info("microphone: %s", self.description)

    def read(self, timeout: float = 1.0) -> AudioChunk | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def read_wav(path: str | Path) -> np.ndarray:
    """16-bit PCM WAV -> mono float32 at 16 kHz."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV files are supported")
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return to_mono_16k(x.reshape(-1, ch) if ch > 1 else x, sr)


def write_wav(path: str | Path, x: np.ndarray) -> None:
    """mono float32 @ 16 kHz -> 16-bit PCM WAV."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(AUDIO_SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def stitch(out: str | Path, parts: list[str | Path], gap_s: float = 1.6) -> None:
    """Concatenate WAV files with silent gaps (used by scripts/make_sample_audio.ps1)."""
    gap = np.zeros(int(AUDIO_SR * gap_s), np.float32)
    write_wav(out, np.concatenate([np.concatenate([gap, read_wav(p)]) for p in parts] + [gap]))


class FileSource(AudioSource):
    """Plays a WAV file in real time (like a microphone would deliver it), optionally looping."""

    name = "file"

    def __init__(self, path: str | Path, loop: bool = True, chunk_ms: int = 100, realtime: bool = True) -> None:
        super().__init__()
        self.path = Path(path)
        self.loop = loop
        self.n = int(AUDIO_SR * chunk_ms / 1000)
        self.realtime = realtime
        self._pos = 0
        self._k = 0
        self._seq = 0

    def start(self) -> None:
        self.audio = read_wav(self.path)
        self.description = f"{self.path.name} ({len(self.audio) / AUDIO_SR:.1f} s{', looping' if self.loop else ''})"
        self._t_start, self._p_start = time.time(), time.perf_counter()
        log.info("audio file: %s", self.description)

    def read(self, timeout: float = 1.0) -> AudioChunk | None:
        if self._pos >= len(self.audio):
            if not self.loop:
                time.sleep(min(timeout, 0.2))
                return None
            self._pos = 0
        if self.realtime:
            wait = self._p_start + (self._k + self.n) / AUDIO_SR - time.perf_counter()
            if wait > timeout:
                time.sleep(timeout)
                return None
            if wait > 0:
                time.sleep(wait)
        x = self.audio[self._pos:self._pos + self.n]
        chunk = AudioChunk(self._t_start + self._k / AUDIO_SR, x, AUDIO_SR, self._seq)
        self._pos += len(x)
        self._k += len(x)
        self._seq += 1
        return chunk

    def stop(self) -> None:
        pass


if __name__ == "__main__":     # python -m bayn.transcription.audio_sources stitch out.wav a.wav b.wav ...
    import sys

    if len(sys.argv) >= 4 and sys.argv[1] == "stitch":
        stitch(sys.argv[2], sys.argv[3:])
    else:
        print("usage: python -m bayn.transcription.audio_sources stitch OUT.wav IN1.wav [IN2.wav ...]")
