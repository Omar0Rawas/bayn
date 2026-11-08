"""Transcription logic without models or microphones: segmenter, audio I/O, language choice, store."""

import numpy as np
import pytest

from bayn.common.config import TranscriptionConfig
from bayn.common.schemas import AUDIO_SR, AudioChunk
from bayn.transcription.audio import level_db
from bayn.transcription.audio_sources import FileSource, read_wav, stitch, to_mono_16k, write_wav
from bayn.transcription.engine import Segment, WhisperEngine, is_hallucination
from bayn.transcription.segmenter import Segmenter
from bayn.transcription.store import TranscriptStore, format_transcript

pytestmark = pytest.mark.unit
SR = AUDIO_SR
T0 = 1_800_000_000.0


def energy_vad(audio: np.ndarray, frame: int = 320, thr: float = 0.02) -> list[dict]:
    """Stand-in for Silero: 20 ms frames with RMS > thr are speech; merge gaps < 300 ms."""
    n = len(audio) // frame
    if n == 0:
        return []
    rms = np.sqrt(np.mean(audio[:n * frame].reshape(n, frame) ** 2, axis=1))
    out, start, gap = [], None, 0
    for i, on in enumerate(rms > thr):
        if on:
            start = i if start is None else start
            gap = 0
        elif start is not None:
            gap += 1
            if gap * frame >= 0.3 * SR:
                out.append({"start": start * frame, "end": (i - gap + 1) * frame})
                start, gap = None, 0
    if start is not None:
        out.append({"start": start * frame, "end": (n - gap) * frame})
    return out


def speech(seconds: float, f: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), np.float32)


def run_stream(seg: Segmenter, audio: np.ndarray, chunk_s: float = 0.1):
    finals, partials, n = [], [], int(chunk_s * SR)
    for i in range(0, len(audio), n):
        seg.feed(audio[i:i + n], T0 + i / SR)
        r = seg.step()
        finals += r.finals
        if r.partial is not None:
            partials.append(r.partial)
    return finals, partials


# ---------------------------------------------------------------- segmenter
def test_utterances_split_on_silence_with_correct_timestamps():
    audio = np.concatenate([silence(1), speech(2), silence(1), speech(1.5), silence(1.5)])
    finals, partials = run_stream(Segmenter(energy_vad, min_silence_s=0.6), audio)
    assert len(finals) == 2
    assert finals[0].t_start == pytest.approx(T0 + 1.0 - 0.2, abs=0.05)      # 0.2 s pad before speech
    assert finals[0].seconds == pytest.approx(2.0 + 0.4, abs=0.1)
    assert finals[1].t_start == pytest.approx(T0 + 4.0 - 0.2, abs=0.05)
    assert partials and all(not p.final for p in partials)                     # live text while speaking


def test_short_pause_does_not_split():
    audio = np.concatenate([speech(1), silence(0.35), speech(1), silence(1.2)])
    finals, _ = run_stream(Segmenter(energy_vad, min_silence_s=0.6), audio)
    assert len(finals) == 1


def test_long_monologue_is_cut():
    audio = np.concatenate([speech(9), silence(0.4), speech(9), silence(1)])
    finals, _ = run_stream(Segmenter(energy_vad, min_silence_s=0.6, max_utterance_s=12), audio)
    assert len(finals) == 2 and finals[0].forced
    assert all(u.seconds <= 12.5 for u in finals)


def test_silence_is_trimmed_and_processed_until_advances():
    seg = Segmenter(energy_vad)
    run_stream(seg, silence(10))
    assert len(seg.buf) <= 1.1 * SR
    assert seg.processed_until() == pytest.approx(T0 + 10 - 0.6, abs=0.15)


def test_processed_until_waits_for_speech_in_progress():
    seg = Segmenter(energy_vad)
    run_stream(seg, np.concatenate([silence(1), speech(2)]))            # still talking
    assert seg.processed_until() == pytest.approx(T0 + 0.8, abs=0.05)


def test_gap_resets_and_flush_finishes_speech():
    seg = Segmenter(energy_vad)
    seg.feed(speech(1), T0)
    seg.feed(speech(1), T0 + 5)                                            # 4 s gap in timestamps
    assert seg.buf_t0 == T0 + 5
    out = seg.flush()
    assert len(out) == 1 and out[0].t_start >= T0 + 5 - 0.21


# ---------------------------------------------------------------- audio I/O
def test_wav_roundtrip_and_resampling(tmp_path):
    x = speech(1.0)
    write_wav(tmp_path / "a.wav", x)
    np.testing.assert_allclose(read_wav(tmp_path / "a.wav"), x, atol=1e-4)
    stereo_44k = np.stack([np.ones(44100), -np.ones(44100)], axis=1) * 0.5
    assert to_mono_16k(stereo_44k, 44100).shape == (16000,)


def test_stitch_and_file_source(tmp_path):
    write_wav(tmp_path / "a.wav", speech(0.5))
    stitch(tmp_path / "s.wav", [tmp_path / "a.wav", tmp_path / "a.wav"], gap_s=0.25)
    assert len(read_wav(tmp_path / "s.wav")) == int(0.25 * SR) * 3 + int(0.5 * SR) * 2
    src = FileSource(tmp_path / "a.wav", loop=True, chunk_ms=100, realtime=False)
    src.start()
    chunks = [src.read() for _ in range(12)]                           # 1.2 s from a 0.5 s file -> loops
    assert all(abs(b.t0 - a.t_end) < 1e-6 for a, b in zip(chunks, chunks[1:]))   # contiguous (float precision)


def test_audio_chunk_int16_roundtrip_and_levels():
    c = AudioChunk(T0, speech(0.1))
    back = AudioChunk.from_msg(c.to_msg())
    np.testing.assert_allclose(back.samples, c.samples, atol=1e-4)
    rms, peak = level_db(speech(0.1))
    assert -15 < rms < -10 and -11 < peak < -10                         # 0.3 sine: -13.5 / -10.5 dBFS
    assert level_db(silence(0.1)) == (-90.0, -90.0)


# ---------------------------------------------------------------- engine logic (no model)
class FakeModel:
    def __init__(self, probs):
        self.probs = probs

    def detect_language(self, audio):
        best = max(self.probs, key=self.probs.get)
        return best, self.probs[best], list(self.probs.items())


def _engine_with(probs) -> WhisperEngine:
    e = WhisperEngine.__new__(WhisperEngine)          # skip model loading
    e.model = FakeModel(probs)
    return e


def test_mixed_mode_only_picks_arabic_or_english():
    # Whisper thinks it's Persian, but in mixed mode we only choose between ar and en
    lang, p = _engine_with({"fa": 0.6, "ar": 0.3, "en": 0.1}).choose_language(np.zeros(16000), "mixed")
    assert lang == "ar" and p == pytest.approx(0.75)
    assert _engine_with({"en": 0.9}).choose_language(np.zeros(1), "ar") == ("ar", 1.0)
    assert _engine_with({"en": 0.9}).choose_language(np.zeros(1), "auto") == (None, 0.0)


def test_hallucination_filter():
    assert is_hallucination(Segment(0, 1, "Thanks for watching!"))
    assert is_hallucination(Segment(0, 1, "اشترك في القناة"))
    assert is_hallucination(Segment(0, 1, " ... "))
    assert is_hallucination(Segment(0, 1, "hello", avg_logprob=-1.5, no_speech_prob=0.9))
    assert not is_hallucination(Segment(0, 1, "He is breathing normally."))


def test_model_auto_choice():
    e = WhisperEngine.__new__(WhisperEngine)
    e.cfg = TranscriptionConfig()
    assert (e._pick_model(True), e._pick_model(False)) == ("large-v3-turbo", "small")
    e.cfg = TranscriptionConfig(model="medium")
    assert e._pick_model(True) == "medium"


# ---------------------------------------------------------------- transcript store
def test_store_window_and_format(tmp_path):
    st = TranscriptStore(tmp_path)
    for i, (a, b) in enumerate([(0, 2), (5, 7), (40, 42)]):
        st.add({"id": f"s{i}", "utterance_id": "u", "t_start": T0 + a, "t_end": T0 + b, "text": f"t{i}",
                "lang": "en" if i % 2 == 0 else "ar", "lang_prob": 1.0, "latency_s": 0.5})
    w = st.window(T0 + 1, T0 + 30)                                     # overlaps first two
    assert [s["id"] for s in w] == ["s0", "s1"]
    assert format_transcript(w, T0 + 5).splitlines() == ["[-005.0 s] (en) t0", "[+000.0 s] (ar) t1"]
    assert [s["id"] for s in st.recent(2)] == ["s1", "s2"]
