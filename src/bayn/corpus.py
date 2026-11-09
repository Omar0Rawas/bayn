"""The recording corpus (tests/data/corpus.json): scripts to read aloud, with key terms and facts.

    python -m bayn.corpus render        # (re)write docs/CORPUS.md from corpus.json
    python -m bayn.corpus list          # ids, languages, recorded or not
    python -m bayn.corpus show mx-pi    # open one script as a big, readable page in the browser
    python -m bayn.corpus show mx-pi --print   # print it instead (fine for English; terminals garble Arabic)

The text in corpus.json is the reference transcript and is never changed for display: the doc and the
reading page only split it into one sentence per line and isolate/highlight the English words inside Arabic
(otherwise the browser's right-to-left layout moves them and their punctuation around).
"""

from __future__ import annotations

import html
import json
import re
import sys
import webbrowser
from functools import lru_cache
from pathlib import Path

from bayn.common.config import DATA_DIR, ROOT

CORPUS = ROOT / "tests" / "data" / "corpus.json"
DOC = ROOT / "docs" / "CORPUS.md"
PAGES = DATA_DIR / "corpus"                                  # generated reading pages (gitignored)
_SENTENCE = re.compile(r"[^.?!؟]+[.?!؟]*")
_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z']*(?: [A-Za-z][A-Za-z']*)*")
LANG_NAMES = {"en": "English only", "ar": "Arabic only", "mixed": "Mixed (Arabic with English key terms)"}
WORDS_PER_SECOND = {"en": 2.4, "ar": 1.8, "mixed": 1.9}     # rough reading speeds, for the time estimate


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def scripts() -> list[dict]:
    return load()["scripts"]


def get(script_id: str) -> dict | None:
    return next((s for s in scripts() if s["id"] == script_id), None)


def topic(script: dict) -> dict:
    return load()["topics"][script["topic"]]


def checks_for(script: dict) -> dict:
    """Summary checks in the format of summarization.quality (facts + forbidden)."""
    t = topic(script)
    return {"facts": t["facts"], "forbidden": t.get("forbidden", {})}


def recording_path(script_id: str) -> Path:
    return DATA_DIR / "recordings" / f"{script_id}.wav"


def est_seconds(script: dict) -> int:
    return round(len(script["text"].split()) / WORDS_PER_SECOND[script["language"]])


def is_rtl(script: dict) -> bool:
    return script["language"] != "en"


def sentences(text: str) -> list[str]:
    """One sentence per line, for reading aloud (joining them with spaces gives the text back)."""
    return [m.group().strip() for m in _SENTENCE.finditer(text) if m.group().strip()]


def sentence_html(sentence: str, rtl: bool) -> str:
    """Escape a sentence; inside Arabic, wrap each English run so it stays in place and stands out."""
    if not rtl:
        return html.escape(sentence)
    parts, last = [], 0
    for m in _LATIN_RUN.finditer(sentence):
        parts += [html.escape(sentence[last:m.start()]), f'<b dir="ltr">{html.escape(m.group())}</b>']
        last = m.end()
    return "".join(parts) + html.escape(sentence[last:])


def reading_page(script: dict) -> str:
    """A self-contained page (no external assets) with the script in large type, one sentence per line."""
    rtl = is_rtl(script)
    lines = "\n".join(f"<p>{sentence_html(s, rtl)}</p>" for s in sentences(script["text"]))
    terms = " · ".join(f"<bdi>{html.escape(k)}</bdi>" for k in script["key_terms"])
    hint = ("English words are <b>bold</b>: say them in English, the way you would in class."
            if script["language"] == "mixed" else "")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(script['id'])} - Bayn corpus</title>
<style>
  :root {{ --bg: #fbfaf7; --fg: #1d1d1f; --muted: #6b6b70; --accent: #0b6e69; --rule: #e3e1dc; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg: #17181a; --fg: #ececec; --muted: #9a9aa0;
    --accent: #5fd0c8; --rule: #2c2d30; }} }}
  body {{ background: var(--bg); color: var(--fg); margin: 0; padding: 32px 16px 64px;
    font-family: "Segoe UI", Tahoma, Arial, sans-serif; }}
  main {{ max-width: 920px; margin: 0 auto; }}
  header {{ color: var(--muted); font-size: 17px; border-bottom: 1px solid var(--rule); padding-bottom: 12px; }}
  header strong {{ color: var(--fg); font-size: 22px; }}
  .script {{ font-size: clamp(24px, 3.2vw, 34px); line-height: 1.75; margin: 28px 0; }}
  .script[dir="rtl"] {{ font-family: "Segoe UI", Tahoma, "Arabic Typesetting", Arial, sans-serif; }}
  .script p {{ margin: 0 0 0.55em; }}
  .script b {{ color: var(--accent); }}
  footer {{ color: var(--muted); font-size: 16px; border-top: 1px solid var(--rule); padding-top: 12px; }}
</style></head>
<body><main>
<header><strong>{html.escape(script['id'])}</strong> · {html.escape(LANG_NAMES[script['language']])} ·
  {html.escape(topic(script)['title'])} · ~{est_seconds(script)} s<br>{hint}</header>
<div class="script" dir="{'rtl' if rtl else 'ltr'}" lang="{'ar' if rtl else 'en'}">
{lines}
</div>
<footer>Key terms that must survive transcription: {terms}<br>
Read naturally, small pauses at the line ends. Press Enter in the terminal when you finish.</footer>
</main></body></html>
"""


def open_reading_page(script: dict) -> Path:
    PAGES.mkdir(parents=True, exist_ok=True)
    page = PAGES / f"{script['id']}.html"
    page.write_text(reading_page(script), encoding="utf-8")
    webbrowser.open(page.resolve().as_uri())
    return page


def print_script(script: dict) -> None:
    """Terminal view. Windows terminals draw Arabic left-to-right and unjoined, so only English is printed."""
    if is_rtl(script):
        print("(Arabic can't be displayed properly in this terminal - read it from the browser page or "
              "docs/CORPUS.md)")
    else:
        print("\n".join(sentences(script["text"])))


def render_markdown() -> str:
    data = load()
    out = ["# Bayn recording corpus",
           "",
           "Scripts to record once and reuse: to compare transcription models on **identical audio** "
           "(`bayn eval-stt --corpus`), to test summaries on realistic transcripts, and as a reference when "
           "you test with a live mic. The same 4 topics in 3 versions: English, Arabic, and mixed (Arabic with "
           "the English key terms a teacher would really use). Names are fictional.",
           "",
           "> Generated from [tests/data/corpus.json](../tests/data/corpus.json) by `python -m bayn.corpus render`. "
           "Edit the JSON, not this file.",
           "",
           "## How to record",
           ""]
    out += [f"{i}. {tip}" for i, tip in enumerate(data["recording_tips"], 1)]
    out += ["", "```powershell", ".\\bayn record mx-pi            # opens the script in big type, Enter stops",
            ".\\bayn corpus show mx-pi       # just open the reading page (e.g. for live-mic tests)",
            ".\\bayn eval-stt --corpus all   # score every recorded script", "```", "",
            "Scripts are shown one sentence per line. In the mixed scripts the **bold** words are English: "
            "say them in English, the way a teacher would in class.", "",
            "| ID | Version | Topic | ~Time |", "|---|---|---|---|"]
    for s in scripts():
        out.append(f"| `{s['id']}` | {LANG_NAMES[s['language']]} | {topic(s)['title']} | {est_seconds(s)} s |")
    for lang in ("en", "ar", "mixed"):
        out += ["", f"## {LANG_NAMES[lang]}", ""]
        for s in [x for x in scripts() if x["language"] == lang]:
            rtl = is_rtl(s)
            out += [f"### `{s['id']}` · {topic(s)['title']} (~{est_seconds(s)} s)", ""]
            # One HTML block (no blank lines inside, so Markdown leaves it alone), one sentence per line.
            body = "<br>\n".join(sentence_html(x, rtl) for x in sentences(s["text"]))
            attrs = 'dir="rtl" lang="ar"' if rtl else 'dir="ltr" lang="en"'
            out += [f"<p {attrs}>\n{body}\n</p>", ""]
            out += ["**Key terms that must survive transcription:** "
                    + ", ".join(f"<bdi>`{k}`</bdi>" if rtl else f"`{k}`" for k in s["key_terms"]), ""]
            if s["topic"] == "incident":
                out += ["*Summary test: a good summary reports what was observed (time, staring, chewing movements, "
                        "about 40 s, didn't fall, tired, nurse called) and does **not** diagnose.*", ""]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = (argv if argv is not None else sys.argv[1:]) or ["list"]
    if args[0] == "render":
        DOC.write_text(render_markdown(), encoding="utf-8")
        print(f"wrote {DOC}")
    elif args[0] == "show" and len(args) > 1 and (s := get(args[1])):
        if "--print" in args:
            print_script(s)
        else:
            print(f"opened {open_reading_page(s)}")
    elif args[0] == "list":
        for s in scripts():
            done = "recorded" if recording_path(s["id"]).exists() else "-"
            print(f"{s['id']:12s} {LANG_NAMES[s['language']]:40s} ~{est_seconds(s):3d} s  {done}")
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
