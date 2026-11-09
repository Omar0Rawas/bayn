"""Word / character error rate for English, Arabic and mixed text.

Normalisation (so only real recognition errors count): Unicode NFKC, Arabic diacritics
and tatweel removed, alef/hamza forms unified (أ إ آ -> ا), ى -> ي, ة -> ه, punctuation
dropped, lower-case. Dialect spelling variants (كدا / كده) still count as errors -
CER is the fairer number for Arabic dialects.

Numbers count the same however they are written, because Whisper often writes digits for spoken
numbers: "ten twenty" = "10-20" = "10:20" = "عشرة وعشرين" -> "10 20"; "forty" = "40" = "٤٠";
"three point one four" = "3.14" = "ثلاثة فاصلة واحد أربعة" -> "3 14".
Key terms may list equally valid spellings separated by "|" (Sara|Sarah, رذرفورد|راذرفورد).
"""

from __future__ import annotations

import re
import unicodedata

_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_TIMESTAMP_PREFIX = re.compile(r"^\s*\[[^\]]*\]\s*(\(\w+\)\s*)?")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "0123456789" * 2)

# number words in normalised spelling (ة -> ه, أ -> ا). Ambiguous dialect words (ست, مية) are left out.
_EN_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                                        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_EN_TENS = {w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)}
_AR_UNITS = {"صفر": 0, "واحد": 1, "واحده": 1, "اثنين": 2, "اثنان": 2, "ثلاث": 3, "ثلاثه": 3, "اربع": 4, "اربعه": 4,
             "خمس": 5, "خمسه": 5, "سته": 6, "سبع": 7, "سبعه": 7, "ثمان": 8, "ثماني": 8, "ثمانيه": 8, "تسع": 9,
             "تسعه": 9, "عشر": 10, "عشره": 10}
_AR_TENS = {f"{stem}{end}": 10 * i for i, stem in enumerate(
    ["عشر", "ثلاث", "اربع", "خمس", "ست", "سبع", "ثمان", "تسع"], 2) for end in ("ين", "ون")}
_NUMBER_WORDS = {**_EN_UNITS, **_EN_TENS, **_AR_UNITS, **_AR_TENS, "hundred": 100, "مئه": 100, "مائه": 100}
_POINT = {"point", "فاصله"}


def _number(word: str) -> int | None:
    if word.isdigit():
        return int(word)
    if word in _NUMBER_WORDS:
        return _NUMBER_WORDS[word]
    if word.startswith("و") and word[1:] in _NUMBER_WORDS:          # وعشرين = "and twenty"
        return _NUMBER_WORDS[word[1:]]
    return None


def _canonical_numbers(words: list[str]) -> list[str]:
    vals = [_number(w) for w in words]
    out: list[str] = []
    i = 0
    while i < len(words):
        v, nxt = vals[i], vals[i + 1] if i + 1 < len(words) else None
        if v is not None and nxt is not None and 1 <= nxt <= 9 and v % 10 == 0 and 20 <= v <= 90 \
                and words[i] in _EN_TENS and words[i + 1] in _EN_UNITS:           # forty two -> 42
            out.append(str(v + nxt)); i += 2; continue
        if v is not None and 1 <= v <= 9 and nxt is not None and words[i + 1].startswith("و") \
                and words[i + 1][1:] in _AR_TENS:                                  # واحد وعشرين -> 21
            out.append(str(v + nxt)); i += 2; continue
        out.append(str(v) if v is not None else words[i])
        i += 1
    # spoken decimals: "3 point 1 4" -> "3 14", like "3.14" (whose dot became a space); digits after the
    # point are said one by one, so they form one number. Other numbers stay separate ("3.4. 3.14").
    joined: list[str] = []
    j = 0
    while j < len(out):
        w = out[j]
        if w in _POINT and joined and joined[-1].isdigit() and j + 1 < len(out) and out[j + 1].isdigit():
            digits = ""
            j += 1
            while j < len(out) and out[j].isdigit() and (not digits or len(out[j]) == 1):
                digits += out[j]
                j += 1
            joined.append(digits)
            continue
        joined.append(w)
        j += 1
    return joined


def normalize(text: str) -> list[str]:
    s = unicodedata.normalize("NFKC", text).translate(_DIGITS)
    s = _DIACRITICS.sub("", s)
    s = re.sub(r"[إأآا]", "ا", s).replace("ى", "ي").replace("ة", "ه").replace("’", "'")
    s = re.sub(r"[^\w\s']", " ", s.lower())
    return _canonical_numbers(s.split())


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
    """Share of key terms (e.g. English terms inside Arabic speech) that came out correctly.
    A term may list alternative spellings: "Sara|Sarah" (reported by its first spelling when missing)."""
    missing = [k.split("|")[0] for k in terms if not any(term_present(a, hypothesis) for a in k.split("|"))]
    return (1 - len(missing) / len(terms) if terms else 1.0), missing


def script_mix(text: str) -> dict[str, float]:
    """Share of Arabic vs Latin letters - shows whether code-switched words kept their script."""
    ar = sum("؀" <= c <= "ۿ" for c in text)
    la = sum(c.isascii() and c.isalpha() for c in text)
    total = max(ar + la, 1)
    return {"arabic": ar / total, "latin": la / total}
