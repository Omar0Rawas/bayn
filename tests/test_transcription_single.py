"""Integration: the single-encode engine copy (engine_single.py) must transcribe exactly like the original.

`mixed` mode normally runs Whisper's encoder twice per utterance (detect_language + transcribe); the copy reuses
the detection pass. Its decoder sees bit-identical input, so whenever both engines pick the same language the
text must be identical. Uses `tiny` on CPU; skipped if it isn't downloaded.
"""

from pathlib import Path

import pytest

from bayn.common.config import TranscriptionConfig
from bayn.transcription.audio_sources import read_wav
from bayn.transcription.engine import WhisperEngine, model_dir, silero_vad
from bayn.transcription.engine_single import SingleEncodeWhisperEngine
from bayn.transcription.regress_single import greedy
from bayn.transcription.segmenter import Segmenter

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (model_dir("tiny") / "model.bin").exists(), reason="run `bayn models` (needs tiny)"),
]
DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def engines():
    # temperature 0 only: faster-whisper's default fallback re-decodes a failed greedy decode with random
    # sampling, which would make even the original engine disagree with itself (tiny on Arabic)
    cfg = TranscriptionConfig(model="tiny", device="cpu", single_encode=True)
    with greedy(WhisperEngine(cfg)) as a, greedy(SingleEncodeWhisperEngine(cfg)) as b:
        yield a, b


@pytest.fixture(scope="module")
def utterances():
    audio = read_wav(Path(__file__).parents[1] / "samples" / "audio" / "demo.wav")
    seg = Segmenter(silero_vad(), min_silence_s=0.6)
    utts = []
    for i in range(0, len(audio), 1600):
        seg.feed(audio[i:i + 1600], 1_800_000_000.0 + i / 16000)
        utts += seg.step().finals
    utts += seg.flush()
    return [read_wav(DATA / "en_short.wav"), read_wav(DATA / "ar_short.wav")] + [u.audio for u in utts]


def test_single_encode_gives_identical_text(engines, utterances):
    original, single = engines
    compared = 0
    for audio in utterances:
        a, b = original.transcribe(audio, "mixed"), single.transcribe(audio, "mixed")
        if a.language == b.language:                       # a borderline ar/en call may flip; nothing else may
            assert b.text == a.text
            compared += 1
    assert compared >= len(utterances) - 1                 # clear speech: at most one coin-flip utterance


def test_one_encoder_pass_per_utterance_and_nothing_leaks(engines, utterances):
    _, single = engines
    before = single.encoder_passes
    for audio in utterances[:3]:
        single.transcribe(audio, "mixed")
        assert single._pending is None                     # never reused for another utterance
    assert single.encoder_passes - before == 3


@pytest.mark.parametrize("flag, cls, active", [("true", SingleEncodeWhisperEngine, True),
                                               ("false", WhisperEngine, False),
                                               ("auto", SingleEncodeWhisperEngine, True)])    # auto + CPU = on
def test_service_uses_the_engine_the_config_asks_for(flag, cls, active, tmp_path):
    from bayn.common.config import load_config
    from bayn.events.store import EventStore
    from bayn.transcription.service import TranscriptionService
    from bayn.transcription.store import TranscriptStore

    cfg = load_config(environ={"BAYN_TRANSCRIPTION__MODEL": "tiny", "BAYN_TRANSCRIPTION__DEVICE": "cpu",
                               "BAYN_TRANSCRIPTION__SINGLE_ENCODE": flag,
                               "BAYN_BUS__STT_PUB_PORT": "0", "BAYN_BUS__STT_CMD_PORT": "0",
                               "BAYN_BUS__AUDIO_PUB_PORT": "59971", "BAYN_BUS__EVENTS_PUB_PORT": "59972"})
    svc = TranscriptionService(cfg, transcripts=TranscriptStore(tmp_path), events=EventStore(tmp_path))
    try:
        assert type(svc.engine) is cls
        assert svc.engine.describe().get("single_encode", False) is active
    finally:
        svc.audio.close(); svc.events_sub.close(); svc.pub.close(); svc.cmd.close()


def test_auto_is_off_on_cuda():
    """`auto` keeps the GPU path unchanged until the dev-PC GPU regression (no GPU needed for this check)."""
    eng = SingleEncodeWhisperEngine.__new__(SingleEncodeWhisperEngine)
    eng.cfg = TranscriptionConfig(model="tiny", device="cpu")                  # single_encode: auto
    eng.model = type("M", (), {"encode": lambda self, f: f})()
    import bayn.transcription.engine as base
    orig = base.WhisperEngine._load
    base.WhisperEngine._load = lambda self, device, name: None                 # skip the real model load
    try:
        eng._load("cuda", "tiny")
        assert eng.single_active is False
        assert eng.model.encode(1) == 1                                        # encoder left untouched
        eng._load("cpu", "tiny")                                               # CUDA failed -> CPU fallback
        assert eng.single_active is True
    finally:
        base.WhisperEngine._load = orig


def test_fixed_language_modes_are_unchanged(engines):
    original, single = engines
    audio = read_wav(DATA / "en_short.wav")
    a, b = original.transcribe(audio, "en"), single.transcribe(audio, "en")
    assert (b.language, b.text) == (a.language, a.text)
