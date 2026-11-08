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


def script_mix(text: str) -> dict[str, float]:
    """Share of Arabic vs Latin letters - shows whether code-switched words kept their script."""
    ar = sum("؀" <= c <= "ۿ" for c in text)
    la = sum(c.isascii() and c.isalpha() for c in text)
    total = max(ar + la, 1)
    return {"arabic": ar / total, "latin": la / total}
