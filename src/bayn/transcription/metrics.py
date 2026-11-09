"""Word / character error rate for English, Arabic and mixed text.

Normalisation (so only real recognition errors count): Unicode NFKC, Arabic diacritics
and tatweel removed, alef/hamza forms unified (أ إ آ -> ا), ى -> ي, ة -> ه, punctuation
dropped, lower-case. Dialect spelling variants (كدا / كده) still count as errors -
CER is the fairer number for Arabic dialects.
"""

from __future__ import annotations

import re
import unicodedata

_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_TIMESTAMP_PREFIX = re.compile(r"^\s*\[[^\]]*\]\s*(\(\w+\)\s*)?")


def normalize(text: str) -> list[str]:
    s = unicodedata.normalize("NFKC", text)
    s = _DIACRITICS.sub("", s)
    s = re.sub(r"[إأآا]", "ا", s).replace("ى", "ي").replace("ة", "ه").replace("’", "'")
    s = re.sub(r"[^\w\s']", " ", s.lower())
    return s.split()


def strip_timestamps(lines: list[str]) -> str:
    """'[+001.7 s] (ar) text' -> 'text' (format of transcript.txt)."""
    return " ".join(_TIMESTAMP_PREFIX.sub("", l) for l in lines)


def _edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def wer(reference: str, hypothesis: str) -> float:
    r, h = normalize(reference), normalize(hypothesis)
    return _edit_distance(r, h) / max(len(r), 1)


def cer(reference: str, hypothesis: str) -> float:
    r, h = "".join(normalize(reference)), "".join(normalize(hypothesis))
    return _edit_distance(list(r), list(h)) / max(len(r), 1)


_AR_PREFIXES = ("", "و", "ب", "ل", "ف", "ك")     # letters Arabic attaches to the front of a word


def term_present(term: str, text: str) -> bool:
    """Is the term in the text (after normalisation)?

    English and short terms: whole words only ('pi' must not match 'piece', 'ط' not every word with ط).
    Longer Arabic terms: also inside a word, because Arabic attaches و/ب/ل/ال (والإلكترونات).
    """
    words = [w[:-2] if w.endswith("'s") else w for w in normalize(text)]     # sun's -> sun
    t, k = " " + " ".join(words) + " ", " ".join(normalize(term))
    if not k:
        return False
    arabic = any("؀" <= c <= "ۿ" for c in k)
    if arabic and len(k) >= 3:
        core = k[2:] if k.startswith("ال") and len(k) > 4 else k
        return core in t
    if arabic:
        return any(f" {p}{k} " in t for p in _AR_PREFIXES)
    singular = k[:-1] if k.endswith("s") and len(k) > 3 else k              # electrons ~ electron
    return any(f" {v} " in t for v in {k, singular, singular + "s"})


def key_term_recall(terms: list[str], hypothesis: str) -> tuple[float, list[str]]:
    """Share of key terms (e.g. English terms inside Arabic speech) that came out correctly."""
    missing = [k for k in terms if not term_present(k, hypothesis)]
    return (1 - len(missing) / len(terms) if terms else 1.0), missing


def script_mix(text: str) -> dict[str, float]:
    """Share of Arabic vs Latin letters - shows whether code-switched words kept their script."""
    ar = sum("؀" <= c <= "ۿ" for c in text)
    la = sum(c.isascii() and c.isalpha() for c in text)
    total = max(ar + la, 1)
    return {"arabic": ar / total, "latin": la / total}
