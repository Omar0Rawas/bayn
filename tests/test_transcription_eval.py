"""WER/CER metrics, decoding strategies, sentence tail, evaluation tool, realistic scenarios."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayn.common.config import TranscriptionConfig
from bayn.transcription.engine import AUTO_PROMPTS, WhisperEngine, model_dir
from bayn.transcription.metrics import cer, normalize, script_mix, strip_timestamps, wer
from bayn.transcription.service import TranscriptionService
from bayn.transcription.store import TranscriptStore

DATA = Path(__file__).parent / "data"
SCENARIOS = json.loads((DATA / "scenarios" / "classroom.json").read_text(encoding="utf-8"))["variants"]
T0 = 1_800_000_000.0


# ---------------------------------------------------------------- metrics
@pytest.mark.unit
def test_normalization_ignores_diacritics_hamza_and_punctuation():
    assert normalize("حرارةُ الشمسِ، أمس!") == normalize("حراره الشمس امس")
    assert normalize("That's right, OK.") == ["that's", "right", "ok"]


@pytest.mark.unit
def test_wer_cer_basics():
    assert wer("a b c d", "a b c d") == 0
    assert wer("a b c d", "a x c") == pytest.approx(0.5)            # 1 substitution + 1 deletion
    assert cer("abcd", "abxd") == pytest.approx(0.25)


@pytest.mark.unit
@pytest.mark.parametrize("variant,cut", [("english", None), ("arabic", "يبرد"), ("mixed", "what happens")])
def test_scores_of_the_live_user_test_are_reproduced(variant, cut):
    """The numbers recorded in TESTING.md come from exactly this scoring."""
    sc = SCENARIOS[variant]
    ref = sc["reference"].rsplit(cut, 1)[0] if cut else sc["reference"]
    hyp = strip_timestamps([l for l in sc["live_transcript"] if not l.startswith("[-")])
    assert wer(ref, hyp) == pytest.approx(sc["scores_2026_11_09"]["wer"], abs=0.001)
    assert cer(ref, hyp) == pytest.approx(sc["scores_2026_11_09"]["cer"], abs=0.001)


@pytest.mark.unit
def test_script_mix_shows_code_switching():
    assert script_mix("وصارت vapor")["latin"] == pytest.approx(5 / 10)
    assert script_mix("shabab shubak")["arabic"] == 0


# ---------------------------------------------------------------- strategies
class _Detect:
    def __init__(self, p_en):
        self.p = {"en": p_en, "ar": 1 - p_en}

    def detect_language(self, audio):
        return max(self.p, key=self.p.get), max(self.p.values()), list(self.p.items())


def _engine(p_en, **cfg):
    e = WhisperEngine.__new__(WhisperEngine)
    e.cfg, e.model = TranscriptionConfig(**cfg), _Detect(p_en)
    return e


@pytest.mark.unit
def test_arabic_lean_keeps_mixed_utterances_in_arabic():
    audio = np.zeros(16000)
    assert _engine(0.7).choose_language(audio, "mixed")[0] == "en"                     # per_utterance
    assert _engine(0.7, mixed_strategy="arabic_lean").choose_language(audio, "mixed")[0] == "ar"
    assert _engine(0.95, mixed_strategy="arabic_lean").choose_language(audio, "mixed")[0] == "en"
    assert _engine(0.7).choose_language(audio, "mixed", strategy="arabic_lean")[0] == "ar"   # override


@pytest.mark.unit
def test_initial_prompt_modes():
    e = _engine(0.5)
    assert e.prompt_for("mixed") is None                                # default: none
    assert e.prompt_for("mixed", "auto") == AUTO_PROMPTS["mixed"]
    assert "دحين" in AUTO_PROMPTS["mixed"] and "think" in AUTO_PROMPTS["mixed"]
    assert e.prompt_for("en", "auto") is None
    assert e.prompt_for("ar", "custom words") == "custom words"


# ---------------------------------------------------------------- sentence tail
@pytest.mark.unit
def test_event_transcript_finishes_the_sentence_in_progress(tmp_path):
    st = TranscriptStore(tmp_path)
    rows = [("a", 1, 4), ("b", 27.5, 29.6), ("c", 30.4, 32.0), ("d", 36.0, 37.0)]   # window ends at +30 s
    for i, a, b in rows:
        st.add({"id": i, "utterance_id": i, "t_start": T0 + a, "t_end": T0 + b, "text": i, "lang": "ar",
                "lang_prob": 1.0, "latency_s": 0.5})
    svc = SimpleNamespace(transcripts=st, cfg=SimpleNamespace(transcription=TranscriptionConfig(event_tail_s=4)))
    segs = TranscriptionService.window_segments(svc, T0 - 10, T0 + 30)
    assert [s["id"] for s in segs] == ["a", "b", "c"]          # c continues b (0.8 s pause); d is a new topic


# ---------------------------------------------------------------- evaluation tool (real models)
@pytest.mark.integration
@pytest.mark.skipif(not (model_dir("tiny") / "model.bin").exists(), reason="run `bayn models` (needs tiny)")
def test_evaluation_tool_scores_strategies_on_a_clip():
    from bayn.transcription.audio_sources import read_wav
    from bayn.transcription.evaluate import STRATEGIES, evaluate

    engine = WhisperEngine(TranscriptionConfig(model="tiny", device="cpu"))
    ref = json.loads((Path(__file__).parents[1] / "samples" / "audio" / "demo_script.json")
                     .read_text(encoding="utf-8"))["en_short"]["text"]
    pick = {k: STRATEGIES[k] for k in ("mixed (current)", "forced en")}
    rows = evaluate(engine, read_wav(DATA / "en_short.wav"), ref, pick)
    assert {r["strategy"] for r in rows} == set(pick)
    assert all(r["wer"] < 0.35 and r["utterances"] >= 1 for r in rows)


# ---------------------------------------------------------------- scenarios fixture
@pytest.mark.unit
def test_classroom_scenarios_are_complete():
    """Realistic scenarios (also used to test summarization in Phase 5)."""
    for name, sc in SCENARIOS.items():
        assert sc["reference"] and sc["live_transcript"], name
        assert all(l.startswith("[") and "s]" in l for l in sc["live_transcript"])
