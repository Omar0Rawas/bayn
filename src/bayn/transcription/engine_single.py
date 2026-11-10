"""Opt-in copy of WhisperEngine that runs Whisper's encoder ONCE per utterance in `mixed` mode.

Why (demo laptop, 2026-11-10, docs/PROGRESS.md): in `mixed` mode the engine first calls detect_language (one
encoder pass) and then transcribe (a second encoder pass over the same audio). On a CPU the encoder is about half
of all Whisper time, and every utterance pays for a full 30 s window however short it is. On the laptop's
power-limited i7-1355U, Whisper small took 2.85 s (detect) + 3.23 s (transcribe) per utterance.

How: language detection runs on exactly the frames that transcribe encodes for the first (only) 30 s window,
and that encoder output is handed to the decoder instead of being computed again. The decoder therefore sees
bit-identical input, so transcripts are identical whenever the ar/en choice is the same. The choice itself can
flip only on coin-flip utterances (P close to 0.5), because detection sees one 10 ms frame fewer than the
original engine.

Status: `transcription.single_encode: auto` (default) = on when Whisper runs on the CPU, off on CUDA; true / false
force it. Regression on the demo laptop's CPU: 0 decode mismatches on base / small / medium / turbo, 26-48 % less
Whisper time (docs/PROGRESS.md). The GPU regression on the dev PC is still pending, hence off on CUDA
(user, 2026-11-10). Compare both engines with:  python -m bayn.transcription.regress_single [--device cuda]
"""

from __future__ import annotations

import numpy as np

from bayn.transcription.engine import MIXED_LANGS, WhisperEngine


class SingleEncodeWhisperEngine(WhisperEngine):
    """WhisperEngine with one encoder pass per utterance in `mixed` mode; other modes are unchanged."""

    def _load(self, device: str, name: str) -> None:
        super()._load(device, name)
        self._pending = None
        self.encoder_passes = 0
        # `single_encode: auto` = only on the CPU (regression-tested there); also re-decided after a CUDA -> CPU
        # fallback, because _load runs again for the CPU
        mode = getattr(self.cfg, "single_encode", True)
        self.single_active = mode is True or (mode == "auto" and device == "cpu")
        if not self.single_active:
            return
        model = self.model
        original = model.encode

        def encode(features):
            # the first encode inside transcribe() is the 30 s window that detection already encoded
            if self._pending is not None:
                out, self._pending = self._pending, None
                return out
            self.encoder_passes += 1
            return original(features)

        model.encode = encode            # instance attribute: generate_segments calls self.encode(...)
        self._encode_once = original

    def describe(self) -> dict:
        return {**super().describe(), "single_encode": self.single_active}

    def choose_language(self, audio: np.ndarray, mode: str,
                        strategy: str | None = None) -> tuple[str | None, float]:
        if mode != "mixed" or not self.single_active:
            return super().choose_language(audio, mode, strategy)
        from faster_whisper.audio import pad_or_trim

        m = self.model
        features = m.feature_extractor(audio)
        # what generate_segments encodes at seek 0: features[:, :min(30 s, content_frames)], content = all but last
        first = features[..., : min(m.feature_extractor.nb_max_frames, features.shape[-1] - 1)]
        enc = self._encode_once(pad_or_trim(first))
        self.encoder_passes += 1
        probs = {tok[2:-2]: p for tok, p in m.model.detect_language(enc)[0]}
        total = sum(probs.get(lang, 0.0) for lang in MIXED_LANGS) or 1.0
        p_en = probs.get("en", 0.0) / total
        strategy = strategy or getattr(self.cfg, "mixed_strategy", "per_utterance")
        if strategy == "arabic_lean":
            need = getattr(self.cfg, "english_confidence", 0.9)
            lang, p = ("en", p_en) if p_en >= need else ("ar", 1 - p_en)
        else:
            lang, p = ("en", p_en) if p_en >= 0.5 else ("ar", 1 - p_en)
        self._pending = enc               # consumed by the next encode() = transcribe's first window
        return lang, p

    def transcribe(self, audio: np.ndarray, mode: str = "mixed", fast: bool = False,
                   strategy: str | None = None, prompt: str | None = None):
        try:
            return super().transcribe(audio, mode, fast, strategy, prompt)
        finally:
            self._pending = None          # never leak an encoder output into another utterance
