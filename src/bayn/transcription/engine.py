"""faster-whisper wrapper: picks model/device for the machine, loads offline, transcribes utterances.

Auto choices (nothing hardcoded per machine):
  NVIDIA GPU usable -> large-v3-turbo, float16, beam 5
  otherwise         -> small, int8 on CPU, beam 1
If loading or a warm-up pass on the GPU fails (missing driver/cuDNN...), it falls back to CPU.
Models live in models/whisper/<name>/ and are downloaded only by `python -m bayn.transcription.models`.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bayn.common.config import MODELS_DIR, TranscriptionConfig

log = logging.getLogger("bayn.stt")

GPU_MODEL, CPU_MODEL = "large-v3-turbo", "small"
MIXED_LANGS = ("ar", "en")

# Style hint for initial_prompt=auto: Whisper continues in the style of its prompt, so a
# code-switched Hejazi/English sample nudges it to keep both scripts and dialect spellings.
AUTO_PROMPTS = {
    "mixed": "طيب يا شباب, eyes on the board. إيش صار؟ The water level نزل, بس where did it go? دحين think about this.",
    "ar": "طيب يا شباب، إيش صار؟ دحين فكروا معايا، فين راحت الموية؟ كدا.",
    "en": None,
    "auto": None,
}

# Phrases Whisper is known to invent on silence / noise (subtitle credits, YouTube outros).
HALLUCINATIONS = (
    "thank you for watching", "thanks for watching", "subscribe", "please like",
    "اشترك في القناة", "ترجمة نانسي قنقر", "شكرا للمشاهدة", "شكراً للمشاهدة",
)


def model_dir(name: str, root: Path | None = None) -> Path:
    return Path(root or MODELS_DIR) / "whisper" / name


def add_cuda_dll_dirs() -> None:
    """Windows: make the pip-installed CUDA libraries (cuBLAS, cuDNN) loadable by CTranslate2."""
    if sys.platform != "win32":
        return
    try:
        import nvidia  # namespace package from nvidia-*-cu12 wheels
    except ImportError:
        return
    for base in getattr(nvidia, "__path__", []):
        for bin_dir in Path(base).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


@dataclass(slots=True)
class Segment:
    start: float          # seconds from the start of the utterance audio
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


@dataclass
class Result:
    language: str
    language_prob: float
    segments: list[Segment] = field(default_factory=list)
    seconds: float = 0.0          # processing time

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments).strip()


def is_hallucination(seg: Segment) -> bool:
    t = seg.text.strip().lower()
    if not t or all(not c.isalnum() for c in t):
        return True
    if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
        return True
    return any(h in t for h in HALLUCINATIONS)


class WhisperEngine:
    def __init__(self, cfg: TranscriptionConfig, models_root: Path | None = None) -> None:
        self.cfg = cfg
        self.models_root = models_root
        add_cuda_dll_dirs()
        want_cuda = cfg.device == "cuda" or (cfg.device == "auto" and cuda_available())
        attempts = [("cuda", self._pick_model(True)), ("cpu", self._pick_model(False))] if want_cuda \
            else [("cpu", self._pick_model(False))]
        errors = []
        for device, name in attempts:
            try:
                self._load(device, name)
                break
            except Exception as e:  # GPU without working CUDA/cuDNN, model missing, ...
                errors.append(f"{device}/{name}: {e}")
                log.warning("could not use %s with %s: %s", name, device, e)
        else:
            raise RuntimeError("no Whisper model could be loaded: " + " | ".join(errors))
        self.fallback_reason = errors[0] if errors else ""

    def _pick_model(self, gpu: bool) -> str:
        if self.cfg.model != "auto":
            return self.cfg.model
        return GPU_MODEL if gpu else CPU_MODEL

    def _load(self, device: str, name: str) -> None:
        from faster_whisper import WhisperModel

        path = model_dir(name, self.models_root)
        if not (path / "model.bin").exists():
            raise FileNotFoundError(f"model '{name}' not downloaded - run: python -m bayn.transcription.models")
        compute = self.cfg.compute_type if self.cfg.compute_type != "auto" else ("float16" if device == "cuda" else "int8")
        t = time.perf_counter()
        threads = self.cfg.cpu_threads if self.cfg.cpu_threads != "auto" else max(1, (os.cpu_count() or 4) - 2)
        model = WhisperModel(str(path), device=device, compute_type=compute,
                             cpu_threads=threads, local_files_only=True)
        # warm-up: CUDA/cuDNN problems often only appear on the first inference
        list(model.transcribe(np.zeros(16000, dtype=np.float32), language="en", beam_size=1)[0])
        self.model, self.device, self.model_name, self.compute_type = model, device, name, compute
        self.beam_size = self.cfg.beam_size if self.cfg.beam_size != "auto" else (5 if device == "cuda" else 1)
        self.load_seconds = time.perf_counter() - t
        log.info("Whisper %s on %s (%s) ready in %.1f s", name, device, compute, self.load_seconds)

    @property
    def partials_enabled(self) -> bool:
        p = self.cfg.partials
        return self.device == "cuda" if p == "auto" else bool(p)

    def choose_language(self, audio: np.ndarray, mode: str,
                        strategy: str | None = None) -> tuple[str | None, float]:
        """Language to force for this utterance (None = let Whisper decide) and its probability."""
        if mode in ("en", "ar"):
            return mode, 1.0
        if mode == "auto":
            return None, 0.0
        # mixed: detect, but only choose between Arabic and English
        _, _, probs = self.model.detect_language(audio)
        p = dict(probs)
        total = sum(p.get(l, 0.0) for l in MIXED_LANGS) or 1.0
        p_en = p.get("en", 0.0) / total
        strategy = strategy or getattr(getattr(self, "cfg", None), "mixed_strategy", "per_utterance")
        if strategy == "arabic_lean":
            # decoding code-switched speech as English translates the Arabic words;
            # decoding it as Arabic keeps the English words -> Arabic unless clearly English
            need = getattr(getattr(self, "cfg", None), "english_confidence", 0.9)
            return ("en", p_en) if p_en >= need else ("ar", 1 - p_en)
        return ("en", p_en) if p_en >= 0.5 else ("ar", 1 - p_en)

    def prompt_for(self, mode: str, prompt: str | None = None) -> str | None:
        p = prompt if prompt is not None else getattr(getattr(self, "cfg", None), "initial_prompt", "none")
        if p in (None, "", "none"):
            return None
        return AUTO_PROMPTS.get(mode) if p == "auto" else p

    def transcribe(self, audio: np.ndarray, mode: str = "mixed", fast: bool = False,
                   strategy: str | None = None, prompt: str | None = None) -> Result:
        """strategy / prompt override the config (used by the evaluation tool)."""
        t = time.perf_counter()
        lang, lang_prob = self.choose_language(audio, mode, strategy)
        segs, info = self.model.transcribe(
            audio, language=lang, beam_size=1 if fast else self.beam_size, vad_filter=False,
            condition_on_previous_text=False, without_timestamps=fast,
            initial_prompt=None if fast else self.prompt_for(mode, prompt))
        out = [Segment(s.start, s.end, s.text.strip(), s.avg_logprob, s.no_speech_prob) for s in segs]
        out = [s for s in out if not is_hallucination(s)]
        return Result(lang or info.language, lang_prob if lang else info.language_probability, out,
                      time.perf_counter() - t)

    def describe(self) -> dict:
        return {"model": self.model_name, "device": self.device, "compute_type": self.compute_type,
                "beam_size": self.beam_size, "partials": self.partials_enabled,
                "load_s": round(self.load_seconds, 1), "fallback": self.fallback_reason}


def silero_vad(threshold: float = 0.5):
    """VAD function for the Segmenter, using the Silero model bundled with faster-whisper."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(threshold=threshold, min_speech_duration_ms=200, min_silence_duration_ms=300,
                      speech_pad_ms=0)
    return lambda audio: get_speech_timestamps(audio, vad_options=opts, sampling_rate=16000)
