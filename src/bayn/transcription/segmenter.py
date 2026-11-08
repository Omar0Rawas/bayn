"""Cut a live audio stream into utterances with voice activity detection (VAD).

Audio is appended as it arrives. Every `step_s` the buffer is checked with the VAD:
  - speech followed by >= min_silence_s of silence  -> a finished utterance
  - speech running longer than max_utterance_s       -> forced cut (at a pause if possible)
  - speech still in progress                         -> optional "partial" for live text
  - no speech                                        -> buffer trimmed
Timestamps are carried along so every utterance knows its unix start time.
The VAD is injectable, so this logic is tested without any model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# vad(audio float32 @ sr) -> list of {"start": sample, "end": sample}
VadFn = Callable[[np.ndarray], list[dict]]

PAD_S = 0.2           # audio kept around speech so words are not clipped
KEEP_IDLE_S = 1.0     # silence kept in the buffer when nobody speaks


@dataclass(slots=True)
class Utterance:
    t_start: float           # unix time of audio[0]
    audio: np.ndarray        # float32 mono
    sr: int
    final: bool = True
    forced: bool = False     # cut because it got too long

    @property
    def t_end(self) -> float:
        return self.t_start + len(self.audio) / self.sr

    @property
    def seconds(self) -> float:
        return len(self.audio) / self.sr


@dataclass(slots=True)
class StepResult:
    finals: list[Utterance]
    partial: Utterance | None
    speaking: bool


class Segmenter:
    def __init__(self, vad: VadFn, sr: int = 16000, min_silence_s: float = 0.6,
                 max_utterance_s: float = 15.0, step_s: float = 0.3) -> None:
        self.vad = vad
        self.sr = sr
        self.min_silence = int(min_silence_s * sr)
        self.max_len = int(max_utterance_s * sr)
        self.step_n = int(step_s * sr)         # run the VAD every this many new samples
        self.pad = int(PAD_S * sr)
        self.reset()

    def reset(self) -> None:
        self.buf = np.zeros(0, dtype=np.float32)
        self.buf_t0: float | None = None
        self._since_step = 0
        self.speech_start_t: float | None = None     # start of speech still in progress

    @property
    def buf_end_t(self) -> float | None:
        return None if self.buf_t0 is None else self.buf_t0 + len(self.buf) / self.sr

    def feed(self, samples: np.ndarray, t0: float) -> None:
        end = self.buf_end_t
        if end is not None and abs(t0 - end) > 0.5:   # gap in the stream: start over
            self.reset()
        if self.buf_t0 is None:
            self.buf_t0 = t0
        self.buf = np.concatenate([self.buf, samples.astype(np.float32)])
        self._since_step += len(samples)

    def _cut(self, start: int, end: int, forced: bool = False) -> Utterance:
        start, end = max(0, start), min(len(self.buf), end)
        return Utterance(self.buf_t0 + start / self.sr, self.buf[start:end].copy(), self.sr, True, forced)

    def _drop(self, n: int) -> None:
        n = max(0, min(n, len(self.buf)))
        self.buf = self.buf[n:]
        self.buf_t0 += n / self.sr

    def step(self, force: bool = False) -> StepResult:
        """Run the VAD if enough new audio arrived (or force=True)."""
        if self.buf_t0 is None or (self._since_step < self.step_n and not force):
            return StepResult([], None, self.speech_start_t is not None)
        self._since_step = 0
        finals: list[Utterance] = []
        while True:
            speech = self.vad(self.buf) if len(self.buf) else []
            if not speech:
                self.speech_start_t = None
                keep = int(KEEP_IDLE_S * self.sr)
                if len(self.buf) > keep:
                    self._drop(len(self.buf) - keep)
                return StepResult(finals, None, False)
            first, last = speech[0], speech[-1]
            silence_after = len(self.buf) - last["end"]
            if silence_after >= self.min_silence:      # utterance finished
                finals.append(self._cut(first["start"] - self.pad, last["end"] + self.pad))
                self._drop(last["end"])
                continue
            if last["end"] - first["start"] >= self.max_len:   # too long: cut at a pause if there is one
                cut = next((s["end"] for s in reversed(speech[:-1])
                            if s["end"] - first["start"] >= self.max_len // 2), None)
                end = cut if cut is not None else len(self.buf)
                finals.append(self._cut(first["start"] - self.pad, end + (self.pad if cut else 0), forced=True))
                self._drop(end)
                continue
            # still speaking
            self.speech_start_t = self.buf_t0 + max(0, first["start"] - self.pad) / self.sr
            partial = self._cut(first["start"] - self.pad, len(self.buf))
            partial.final = False
            return StepResult(finals, partial, True)

    def flush(self) -> list[Utterance]:
        """End of stream: whatever speech is buffered becomes a final utterance."""
        speech = self.vad(self.buf) if self.buf_t0 is not None and len(self.buf) else []
        out = [self._cut(speech[0]["start"] - self.pad, speech[-1]["end"] + self.pad)] if speech else []
        self.reset()
        return out

    def processed_until(self) -> float | None:
        """All audio before this unix time has been turned into utterances (or was silence)."""
        if self.speech_start_t is not None:
            return self.speech_start_t
        end = self.buf_end_t
        return None if end is None else end - self.min_silence / self.sr
