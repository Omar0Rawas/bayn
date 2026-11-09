"""Summaries: prompts, structured rendering, quality checks, model choice, Ollama client, service."""

import json
from pathlib import Path

import pytest

from bayn.common.config import AppConfig, SummaryConfig
from bayn.events.store import EventStore
from bayn.summarization import models as summary_models
from bayn.summarization.ollama_client import OllamaClient, OllamaError
from bayn.summarization.prompts import (assemble, build_messages, detect_language, eeg_section, output_languages,
                                        partial_render, render, schema)
from bayn.summarization.quality import check_summary
from tests.fake_ollama import ANSWERS, FakeOllama

DATA = json.loads((Path(__file__).parent / "data" / "scenarios" / "classroom.json").read_text(encoding="utf-8"))
CHECKS = DATA["summary_checks"]
T = 1_800_000_000.0
EVENT = {"id": "evt-20270115-080000-000", "t": T, "detector": "clench", "source": "gesture", "label": "Jaw clench",
         "pre_s": 10, "post_s": 30}
SEGS = [{"t_start": T + 1, "t_end": T + 3, "text": "okay everyone eyes on the board", "lang": "en"},
        {"t_start": T + 5, "t_end": T + 8, "text": "the water level went down", "lang": "en"}]
AR_SEGS = [{"t_start": T + 4, "t_end": T + 6, "text": "امس حطينا كاسة موية جنب الشباك", "lang": "ar"}]
STATS = {"before": {"dominant": {"TP9": "alpha", "AF7": "delta"}},
         "after": {"dominant": {"TP9": "beta", "AF7": "delta"}, "quality": {"TP9": "good", "AF7": "ok"}}}


# ---------------------------------------------------------------- prompts (unit)
@pytest.mark.unit
def test_output_language_follows_the_transcript():
    assert detect_language(SEGS) == "en" and detect_language(AR_SEGS) == "ar"
    assert output_languages("auto", AR_SEGS) == ["ar"]
    assert output_languages("both", SEGS) == ["en", "ar"]
    assert output_languages("en", AR_SEGS) == ["en"]


@pytest.mark.unit
def test_prompt_has_rules_json_example_and_relative_times():
    msgs = build_messages(EVENT, SEGS, "en", max_points=3)
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert "ONLY the transcript" in system and "not a medical assessment" in system
    assert '"key_points"' in system and "at most 3" in system and "museum" in system      # unrelated example
    assert "[+1 s] okay everyone" in user and "jaw-clench gesture" in user and "as JSON" in user


@pytest.mark.unit
def test_arabic_prompt_is_written_in_arabic():
    system, user = (m["content"] for m in build_messages(EVENT, AR_SEGS, "ar"))
    assert "اكتب باللغة العربية" in system and "المصطلحات الإنجليزية" in system
    assert "بصيغة JSON" in user


@pytest.mark.unit
def test_schema_caps_key_points():
    assert schema(4)["properties"]["key_points"]["maxItems"] == 4


@pytest.mark.unit
def test_render_localised_headings():
    md_en = render(ANSWERS["en"], "en")
    assert md_en.startswith("**Summary:**") and "**Key points:**" in md_en and md_en.endswith("**Unclear:** none")
    md_ar = render(ANSWERS["ar"], "ar")
    assert "**الملخص:**" in md_ar and "**غير واضح:** فرحت" in md_ar


@pytest.mark.unit
def test_partial_render_streams_text_while_json_is_incomplete():
    full = json.dumps(ANSWERS["en"])
    halfway = full[: full.index('"key_points"') + 60]
    md = partial_render(halfway, "en")
    assert md.startswith("**Summary:** A teacher explains") and "**Key points:**" not in md or "cup" in md
    assert partial_render('{"summ', "en") == ""                               # nothing to show yet
    assert partial_render(full, "en") == render(ANSWERS["en"], "en")           # complete = final


@pytest.mark.unit
def test_eeg_section_is_written_by_code_not_the_llm():
    s = eeg_section(STATS, "en")
    assert "not a medical assessment" in s and "TP9 alpha → beta" in s and "AF7 ok" in s
    assert "ألفا ← " not in eeg_section(STATS, "ar") and "TP9 ألفا → بيتا" in eeg_section(STATS, "ar")
    assert "no EEG data" in eeg_section(None, "en")


@pytest.mark.unit
def test_assembled_document_has_disclaimer():
    doc = assemble(EVENT, render(ANSWERS["en"], "en"), STATS, "en", "qwen2.5:7b")
    assert doc.startswith("### Event: Jaw clench") and "not a medical device" in doc and "qwen2.5:7b" in doc


# ---------------------------------------------------------------- quality checks (unit)
@pytest.mark.unit
def test_quality_check_rewards_facts_and_catches_flips_and_wrong_language():
    good = check_summary(render(ANSWERS["en"], "en"), "en", CHECKS, "english")
    assert good.coverage >= 0.8 and not good.invented and good.language_ok and good.format_ok
    flipped = check_summary("**Summary:** The water level rose. **Key points:** x **Unclear:** none", "en", CHECKS)
    assert flipped.invented == ["level rose"]
    wrong_lang = check_summary(render(ANSWERS["en"], "en"), "ar", CHECKS, "arabic")
    assert not wrong_lang.language_ok
    ar = check_summary(render(ANSWERS["ar"], "ar"), "ar", CHECKS, "arabic")
    assert ar.language_ok and ar.format_ok and "cup of water by the window" in ar.facts_found


@pytest.mark.unit
def test_facts_only_required_where_the_transcript_has_them():
    report = check_summary(render(ANSWERS["en"], "en"), "en", CHECKS, "mixed")
    assert len(report.facts_found) + len(report.facts_missing) == 2


# ---------------------------------------------------------------- model choice (unit)
@pytest.mark.unit
def test_auto_model_depends_on_gpu_and_ram(monkeypatch):
    monkeypatch.setattr(summary_models, "apple_silicon_memory_gb", lambda: 0.0)
    monkeypatch.setattr(summary_models, "system_ram_gb", lambda: 8.0)
    monkeypatch.setattr(summary_models, "gpu_memory_mb", lambda: 11264)
    assert summary_models.resolve_model(SummaryConfig()) == "gemma4:e2b"
    monkeypatch.setattr(summary_models, "gpu_memory_mb", lambda: 6144)        # 6 GB laptop GPU: fits with Whisper small
    assert summary_models.resolve_model(SummaryConfig()) == "gemma4:e2b"
    monkeypatch.setattr(summary_models, "gpu_memory_mb", lambda: 4096)        # small GPU, 8 GB RAM: the light model
    assert summary_models.resolve_model(SummaryConfig()) == "qwen3:4b"
    monkeypatch.setattr(summary_models, "gpu_memory_mb", lambda: 0)           # CPU only
    assert summary_models.resolve_model(SummaryConfig()) == "qwen3:4b"
    monkeypatch.setattr(summary_models, "system_ram_gb", lambda: 16.0)        # CPU only, 16 GB RAM
    assert summary_models.resolve_model(SummaryConfig()) == "gemma4:e2b"
    assert summary_models.resolve_model(SummaryConfig(), available=["qwen2.5:7b"]) == "qwen2.5:7b"   # use what's there


@pytest.mark.unit
def test_system_ram_is_detected():
    assert 1 < summary_models.system_ram_gb() < 4096
    assert summary_models.resolve_model(SummaryConfig(model="llama3.2:3b")) == "llama3.2:3b"


# ---------------------------------------------------------------- Ollama client + service (integration, fake server)
@pytest.fixture
def fake():
    f = FakeOllama()
    yield f
    f.close()


@pytest.mark.integration
def test_client_streams_tokens_and_reports_speed(fake):
    c = OllamaClient(fake.url)
    assert c.version() == "fake-0.1" and c.has_model("qwen2.5:3b")
    pieces = []
    r = c.chat("qwen2.5:3b", build_messages(EVENT, SEGS, "en"), on_token=pieces.append, json_schema=schema(4))
    assert len(pieces) > 5 and json.loads(r.text) == ANSWERS["en"] and r.tokens_per_s == 100
    assert fake.requests[-1]["format"]["required"] == ["summary", "key_points", "unclear"]


@pytest.mark.integration
def test_client_error_and_unreachable():
    f = FakeOllama(fail_chat="prediction aborted, token repeat limit reached")
    try:
        with pytest.raises(OllamaError, match="repeat limit"):
            OllamaClient(f.url).chat("qwen2.5:3b", [{"role": "user", "content": "x"}])
    finally:
        f.close()
    assert OllamaClient("http://127.0.0.1:9").version() is None


@pytest.mark.integration
def test_stalled_model_times_out_instead_of_hanging(fake, monkeypatch):
    """Regression (D17): a model that stops producing tokens must not block summaries forever."""
    import http.server

    class Stall(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.send_response(200)
            self.end_headers()
            import time as _t
            _t.sleep(3)                                  # never sends a token

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stall)
    import threading
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = OllamaClient(f"http://127.0.0.1:{srv.server_address[1]}", stall_s=0.5)
        with pytest.raises(OllamaError, match="stopped responding"):
            c.chat("x", [{"role": "user", "content": "x"}])
    finally:
        srv.shutdown()


def _service(fake, tmp_path, monkeypatch, **summary):
    from bayn.summarization.service import SummaryService

    cfg = AppConfig.model_validate({"summary": {"host": fake.url, "model": "qwen2.5:3b", **summary},
                                    "bus": {"sum_pub_port": 0, "sum_cmd_port": 0, "stt_pub_port": 0}})
    store = EventStore(tmp_path)
    store.upsert({**EVENT, "kind": "gesture", "state": "saved", "stats": STATS})
    svc = SummaryService(cfg, events=store)
    sent = []
    monkeypatch.setattr(svc.pub, "send", lambda topic, payload: sent.append((topic, payload)))
    assert svc._ready.wait(10)
    return svc, store, sent


@pytest.mark.integration
def test_service_summarises_window_streams_and_saves(fake, tmp_path, monkeypatch):
    svc, store, sent = _service(fake, tmp_path, monkeypatch, language="both")
    window = {"event_id": EVENT["id"], "t_event": T, "segments": SEGS}
    done = svc.summarize(window)
    assert [d["lang"] for d in done] == ["en", "ar"] and all(not d["error"] for d in done)
    tokens = [p for t, p in sent if t == "summary.token" and p.get("text")]
    assert tokens and tokens[0]["text"].startswith("**Summary:**")              # streamed while writing
    md = (store.dir_for(EVENT["id"]) / "summary.md").read_text(encoding="utf-8")
    assert "**Summary:**" in md and "**الملخص:**" in md and "TP9 alpha → beta" in md
    meta = store.get(EVENT["id"])
    assert meta["summary"]["languages"] == ["en", "ar"] and "summary.md" in meta["files"]
    assert done[0]["prompt"][0]["role"] == "system"                              # prompt shown in the UI


@pytest.mark.integration
def test_service_handles_no_speech_and_llm_failure(fake, tmp_path, monkeypatch):
    svc, _, _ = _service(fake, tmp_path, monkeypatch)
    (msg,) = svc.summarize({"event_id": EVENT["id"], "t_event": T, "segments": []})
    assert "No speech" in msg["markdown"] and msg["prompt"] is None              # no LLM call at all
    fake.fail_chat = "prediction aborted, token repeat limit reached"
    (msg,) = svc.summarize({"event_id": EVENT["id"], "t_event": T, "segments": SEGS})
    assert "repeat limit" in msg["error"]                                       # reported, not crashed
    assert sum(r["path"] == "/api/chat" for r in fake.requests) == 2            # one retry


# ---------------------------------------------------------------- real model (integration, needs Ollama)
def _real_model():
    c = OllamaClient()
    if c.version() is None:
        return None
    m = summary_models.resolve_model(SummaryConfig(), c.models())
    return m if c.has_model(m) else None


@pytest.mark.integration
@pytest.mark.skipif(_real_model() is None, reason="Ollama with the summary model not available")
def test_real_model_summarises_the_english_classroom_scenario():
    from bayn.summarization.evaluate import scenario_segments

    model = _real_model()
    segs = scenario_segments(DATA["variants"]["english"]["live_transcript"])
    r = OllamaClient().chat(model, build_messages({**EVENT, "detector": "manual"}, segs, "en"), json_schema=schema(4))
    q = check_summary(render(json.loads(r.text), "en"), "en", CHECKS, "english")
    assert q.coverage >= 0.6 and not q.invented and q.language_ok and q.format_ok, q
