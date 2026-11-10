"""Model benchmark (bayn.benchmark): fair number scoring, the LLM stage against a fake Ollama,
CPU-only options, and the report / per-machine recommendation from cached results."""

import argparse
import json

import pytest

from bayn import benchmark as bench
from bayn import corpus
from bayn.summarization.ollama_client import OllamaClient
from bayn.transcription.metrics import key_term_recall, normalize, wer
from tests.fake_ollama import FakeOllama


@pytest.mark.unit
@pytest.mark.parametrize("spoken, written", [
    ("It's ten twenty", "It's 10-20"), ("ten twenty", "10:20"), ("forty seconds", "40 seconds"),
    ("about three point one four", "about 3.14."),
    ("three point one four", "3.14"), ("forty two", "42"),
    ("الساعة عشرة وعشرين", "الساعة 10:20"), ("ثلاثة فاصلة واحد أربعة", "٣٫١٤"), ("أربعين ثانية", "40 ثانية"),
])
def test_numbers_score_the_same_in_words_and_digits(spoken, written):
    assert normalize(spoken) == normalize(written)
    assert wer(spoken, written) == 0


@pytest.mark.unit
def test_digit_transcripts_keep_the_key_terms():
    en = corpus.get("en-incident")
    recall, missing = key_term_recall(en["key_terms"], "It's 10-20. Sara is staring, chewing, about 40 seconds. "
                                                       "Call the nurse.")
    assert recall == 1.0, missing
    assert normalize("three round objects") == ["3", "round", "objects"]       # both sides map the same way
    assert normalize("about 3.4. 3.14") == ["about", "3", "4", "3", "14"]       # a self-correction stays visible
    assert key_term_recall(["three point one four"], "about 3.4. 3.14 we call")[0] == 1.0
    assert key_term_recall(["Sara|Sarah"], "Sarah stopped answering") == (1.0, [])
    assert key_term_recall(["Sara|Sarah"], "She stopped") == (0.0, ["Sara"])


@pytest.mark.unit
def test_reference_source_is_the_script_sentence_by_sentence(monkeypatch):
    monkeypatch.setattr(bench, "recorded", lambda: corpus.scripts())
    segs = bench.source_segments("reference")
    assert set(segs) == {s["id"] for s in corpus.scripts()}
    mx = segs["mx-pi"]
    assert len(mx) >= 4 and all(s["lang"] == "ar" for s in mx)
    assert " ".join(s["text"] for s in mx) == " ".join(corpus.get("mx-pi")["text"].split())


@pytest.mark.unit
def test_memory_probe_reports_this_process():
    assert bench.peak_rss_mb() > 10


@pytest.mark.integration
def test_chat_asks_not_to_think_and_retries_without_the_switch():
    fake = FakeOllama(rejects_think=True)
    try:
        r = OllamaClient(fake.url).chat("qwen2.5:3b", [{"role": "system", "content": "x"}],
                                        options={"num_gpu": 0, "num_thread": 4})
        assert r.text
        chats = [q for q in fake.requests if q["path"] == "/api/chat"]
        assert chats[0]["think"] is False and "think" not in chats[1]          # retried without it
        assert chats[1]["options"]["num_gpu"] == 0 and chats[1]["options"]["num_thread"] == 4
    finally:
        fake.close()


@pytest.fixture
def bench_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "BENCH", tmp_path)
    monkeypatch.setattr(bench, "STT_DIR", tmp_path / "stt")
    monkeypatch.setattr(bench, "LLM_DIR", tmp_path / "llm")
    scripts = [corpus.get(i) for i in ("en-incident", "mx-pi")]
    monkeypatch.setattr(bench, "recorded", lambda: scripts)
    return tmp_path


def _fake_stt(profile, model, rtf, vram, ram):
    scripts = {}
    for s in bench.recorded():
        segs = bench.source_segments("reference")[s["id"]]
        scripts[s["id"]] = {"version": s["language"], "rtf": rtf, "segments": segs,
                            "text": " ".join(x["text"] for x in segs)}
    bench._save(bench.stt_path(profile, model), {"model": model, "profile": profile, "scripts": scripts,
                                                  "engine": {"load_s": 1.0}, "vram_mb": vram, "ram_mb": ram,
                                                  "machine": {"cpu": "test", "cpu_count": 8, "gpu": ""}})


@pytest.mark.integration
def test_llm_stage_report_and_recommendation(bench_dirs, monkeypatch):
    fake = FakeOllama(models=("qwen2.5:3b",))
    monkeypatch.setenv("BAYN_SUMMARY__HOST", fake.url)
    try:
        _fake_stt("gpu", "small", rtf=0.05, vram=1000, ram=800)
        _fake_stt("laptop", "small", rtf=0.3, vram=None, ram=900)
        run = lambda **kw: bench.run_llm(argparse.Namespace(
            **{"models": ["qwen2.5:3b"], "sources": ["all"], "profile": "gpu", "scripts": None,
               "langs": ["en", "ar"], "force": False, **kw}))
        assert run() == 0
        assert run(profile="laptop", sources=["reference"], langs=["en"]) == 0
        cached = json.loads(bench.llm_path("gpu", "qwen2.5:3b", "gpu/small").read_text(encoding="utf-8"))
        assert len(cached["cases"]) == 4 and cached["memory"]["vram_mb"] == 2048
        assert all("quality" in c for c in cached["cases"].values())
        lap = [q for q in fake.requests if q["path"] == "/api/chat" and q["options"].get("num_gpu") == 0]
        assert len(lap) == 2                                         # laptop profile = CPU only
        n = len([q for q in fake.requests if q["path"] == "/api/chat"])
        assert run() == 0 and len([q for q in fake.requests if q["path"] == "/api/chat"]) == n   # cached: no re-run
    finally:
        fake.close()
    md = bench.render_report()
    for heading in ("## 1. Transcription", "## 2. Summaries from a perfect transcript", "## 3. Whisper × LLM",
                    "## 4. Summary speed without a GPU", "## 5. Best measured combination"):
        assert heading in md
    rec = {t: (w, m) for t, w, m, _ in bench.recommend(*bench.load_results())}
    assert rec["NVIDIA GPU with 4 GB"] == ("small", "qwen2.5:3b")          # 1000 MB + 2048 MB fits 3300
    assert rec["CPU only, 8 GB RAM"] == ("small", "qwen2.5:3b")            # 900 MB + 3072 MB fits 4500


@pytest.mark.integration
def test_report_on_a_machine_without_gpu_uses_the_cpu_runs(bench_dirs, monkeypatch):
    """Demo laptop: no gpu profile at all, so summary quality (sections 2, 3, 5) comes from the cpu runs."""
    fake = FakeOllama(models=("qwen2.5:3b",))
    monkeypatch.setenv("BAYN_SUMMARY__HOST", fake.url)
    try:
        _fake_stt("cpu", "small", rtf=0.3, vram=None, ram=600)
        _fake_stt("cpu", "medium", rtf=1.5, vram=None, ram=2000)          # too slow for a CPU recommendation
        assert bench.run_llm(argparse.Namespace(models=["qwen2.5:3b"], sources=["all"], profile="cpu",
                                                scripts=None, langs=["en"], force=False)) == 0
    finally:
        fake.close()
    stt, llm = bench.load_results()
    assert bench.quality_profile(llm) == "cpu"
    md = bench.render_report()
    assert "dev PC" not in md and "## 3. Whisper × LLM" in md
    sec2 = md.split("## 2.")[1].split("## 3.")[0]
    assert "| qwen2.5:3b |" in sec2 and "Time (cpu)" in sec2
    assert "| 6 |" in md.split("## 4.")[1]                                 # speed over all 3 sources × 2 scripts
    rec = {t: (w, m) for t, w, m, _ in bench.recommend(stt, llm)}
    assert rec["CPU only, 8 GB RAM"] == ("small", "qwen2.5:3b")
    assert rec["NVIDIA GPU with 4 GB"][0] == "–"
