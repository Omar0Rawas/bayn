"""The recording corpus is self-consistent: every script contains its own key terms and its topic's
facts (so a perfect transcript/summary scores 100 %), nothing forbidden, and docs/CORPUS.md is current."""

import pytest

from bayn import corpus
from bayn.summarization.quality import check_summary
from bayn.transcription.metrics import key_term_recall, term_present

pytestmark = pytest.mark.unit
SCRIPTS = corpus.scripts()


def test_corpus_shape():
    langs = {s["language"] for s in SCRIPTS}
    topics = {s["topic"] for s in SCRIPTS}
    assert langs == {"en", "ar", "mixed"}
    for t in topics:                                    # parallel corpus: every topic in every version
        assert {s["language"] for s in SCRIPTS if s["topic"] == t} == langs, t
    assert len({s["id"] for s in SCRIPTS}) == len(SCRIPTS)
    assert all(20 <= corpus.est_seconds(s) <= 60 for s in SCRIPTS)


@pytest.mark.parametrize("script", SCRIPTS, ids=[s["id"] for s in SCRIPTS])
def test_script_contains_its_key_terms(script):
    recall, missing = key_term_recall(script["key_terms"], script["text"])
    assert recall == 1.0, missing


@pytest.mark.parametrize("script", SCRIPTS, ids=[s["id"] for s in SCRIPTS])
def test_reading_the_script_back_covers_every_fact_and_nothing_forbidden(script):
    lang = "en" if script["language"] == "en" else "ar"     # mixed scripts are checked with the Arabic keywords
    report = check_summary(script["text"], lang, corpus.checks_for(script))
    assert report.facts_missing == [] and report.invented == [], report


def test_mixed_scripts_keep_english_terms_in_latin_script():
    for s in SCRIPTS:
        if s["language"] == "mixed":
            assert all(t.isascii() for t in s["key_terms"]), s["id"]


def test_term_matching_rules():
    assert term_present("pi", "we call this number pi.") and not term_present("pi", "a piece of paper")
    assert term_present("ط", "نسمي هذا العدد ط.") and not term_present("ط", "طيب يا شباب")
    assert term_present("الإلكترونات", "والإلكترونات تتحرك حولها")          # attached و
    assert term_present("electrons", "the electron moves") and term_present("sun", "the sun's heat")


@pytest.mark.parametrize("script", SCRIPTS, ids=[s["id"] for s in SCRIPTS])
def test_display_keeps_the_text(script):
    """One sentence per line is display only: joining the lines gives the reference text back."""
    lines = corpus.sentences(script["text"])
    assert len(lines) >= 4
    assert " ".join(lines) == " ".join(script["text"].split())


def test_english_inside_arabic_is_isolated_and_page_is_offline():
    mx = corpus.get("mx-pi")
    line = corpus.sentence_html("إذن الـ formula: الـ circumference يساوي pi ضرب الـ diameter.", rtl=True)
    assert '<b dir="ltr">formula</b>:' in line and '<b dir="ltr">pi</b>' in line
    assert '<b dir="ltr">three point one four</b>' in corpus.reading_page(mx)     # multi-word runs stay together
    assert corpus.sentence_html("It's ten twenty.", rtl=False) == "It&#x27;s ten twenty."
    page = corpus.reading_page(mx)
    assert 'dir="rtl"' in page and "http" not in page                           # no CDN / external assets


def test_generated_doc_is_up_to_date():
    assert corpus.DOC.read_text(encoding="utf-8") == corpus.render_markdown(), \
        "docs/CORPUS.md is stale: run  python -m bayn.corpus render"
