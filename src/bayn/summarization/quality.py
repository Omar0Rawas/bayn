"""Automatic quality checks for a summary against a scenario's expectations.

Not a substitute for reading the summary, but it makes model / prompt choices comparable:
  coverage   - share of expected key facts present (keyword groups, both languages)
  invented   - forbidden content found (recognition errors stated as facts, chatter, medical claims)
  language   - the summary is written in the requested language
  format     - the expected headings are there
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bayn.transcription.metrics import normalize, script_mix, term_present

HEADINGS = {"en": ("summary", "key points", "unclear"), "ar": ("الملخص", "النقاط الرئيسية", "غير واضح")}


def _norm(text: str) -> str:
    return " ".join(normalize(text))


@dataclass
class QualityReport:
    coverage: float
    facts_found: list[str]
    facts_missing: list[str]
    invented: list[str]
    language_ok: bool
    arabic_share: float
    format_ok: bool
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.coverage >= 0.8 and not self.invented and self.language_ok and self.format_ok

    def to_dict(self) -> dict:
        return {**self.__dict__, "passed": self.passed}


def applicable_facts(checks: dict, variant: str | None) -> list[dict]:
    """Facts that the variant's transcript actually contains (a summary can't know the rest)."""
    return [f for f in checks["facts"] if variant is None or variant in f.get("in", [variant])]


def check_summary(text: str, lang: str, checks: dict, variant: str | None = None) -> QualityReport:
    t = _norm(text)
    facts = applicable_facts(checks, variant)
    found, missing = [], []
    for fact in facts:
        groups = fact.get(lang) or fact["en"]
        # very short keywords ("pi", "ط", "40") must be whole words; longer ones may be part of a word
        ok = all(any(term_present(k, text) if len(_norm(k)) <= 2 else _norm(k) in t for k in group)
                 for group in groups)
        (found if ok else missing).append(fact["name"])
    invented = [w for w in checks.get("forbidden", {}).get(lang, []) if not w.startswith("_") and _norm(w) in t]
    share = script_mix(text)["arabic"]
    language_ok = share >= 0.75 if lang == "ar" else share <= 0.05
    format_ok = all(_norm(h) in t for h in HEADINGS[lang])
    return QualityReport(len(found) / max(len(facts), 1), found, missing, invented,
                         language_ok, round(share, 2), format_ok)
