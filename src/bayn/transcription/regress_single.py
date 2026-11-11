"""Regression run: original WhisperEngine vs SingleEncodeWhisperEngine (engine_single.py) on the same utterances.

    python -m bayn.transcription.regress_single                       # this machine's auto model and device
    python -m bayn.transcription.regress_single --models small base --device cpu
    python -m bayn.transcription.regress_single --models large-v3-turbo --device cuda   # dev PC

Input: the corpus recordings (data/recordings, `bayn record`), segmented exactly like `bayn benchmark stt`;
without recordings, the test clips (tests/data/*.wav, samples/audio/demo.wav).
Per utterance it compares the language chosen and the text:
  - different language = language flip on a borderline utterance (reported with both probabilities);
  - same language, different text: both engines re-decode it at temperature 0. Identical then = "sampling"
    (faster-whisper's random temperature fallback on a failed greedy decode; the original engine also
    disagrees with itself there). Still different = DECODE MISMATCH: must never happen (exit code 1).
Also reports speed (RTF, seconds per utterance) and encoder passes per utterance (should be 1.0 vs 2.0).
Results -> data/eval/regress/<model>-<device>.json (gitignored).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from contextlib import contextmanager

from bayn import corpus
from bayn.common.config import DATA_DIR, ROOT, load_config
from bayn.transcription.audio_sources import read_wav
from bayn.transcription.engine import WhisperEngine
from bayn.transcription.engine_single import SingleEncodeWhisperEngine
from bayn.transcription.evaluate import segment
from bayn.transcription.metrics import wer

OUT = DATA_DIR / "eval" / "regress"


def clips() -> dict[str, tuple[object, str | None]]:
    """id -> (16 kHz audio, reference text or None)."""
    rec = {s["id"]: (read_wav(corpus.recording_path(s["id"])), s["text"])
           for s in corpus.scripts() if corpus.recording_path(s["id"]).exists()}
    if rec:
        return rec
    files = sorted((ROOT / "tests" / "data").glob("*.wav")) + [ROOT / "samples" / "audio" / "demo.wav"]
    return {p.stem: (read_wav(p), None) for p in files if p.exists()}


@contextmanager
def greedy(engine):
    """Decode at temperature 0 only. By default faster-whisper retries a decode that fails its quality checks
    (garbled Arabic on small models) with random sampling, so even the original engine disagrees with itself
    there (laptop, 2026-11-10: 3 runs -> 3 different texts). At T=0 decoding is deterministic."""
    orig = engine.model.transcribe
    engine.model.transcribe = lambda *a, **k: orig(*a, **{**k, "temperature": 0.0})
    try:
        yield engine
    finally:
        engine.model.transcribe = orig


def same_greedy(a, b, audio, mode: str) -> bool:
    """Re-decode one utterance with both engines at T=0: identical text = the difference was only sampling."""
    with greedy(a), greedy(b):
        ra, rb = a.transcribe(audio, mode), b.transcribe(audio, mode)
    return ra.language == rb.language and ra.text == rb.text


def run(engine, utts: dict, mode: str) -> dict:
    out = {}
    for cid, us in utts.items():
        rows = []
        for u in us:
            t = time.perf_counter()
            r = engine.transcribe(u.audio, mode)
            rows.append({"lang": r.language, "p": round(r.language_prob, 3), "text": r.text,
                         "s": time.perf_counter() - t})
        out[cid] = rows
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Original vs single-encode Whisper engine on the same utterances")
    ap.add_argument("--models", nargs="*", default=["auto"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    data = clips()
    utts = {cid: segment(a, cfg.transcription.min_silence_s) for cid, (a, _) in data.items()}
    n_utts = sum(len(u) for u in utts.values())
    audio_s = sum(len(a) for a, _ in data.values()) / 16000
    print(f"{len(data)} clips, {n_utts} utterances, {audio_s:.0f} s of audio, language mode "
          f"{cfg.transcription.language}")
    if cfg.transcription.language != "mixed":
        print("note: the engines only differ in `mixed` mode")
    failed = False
    for model in args.models:
        # single_encode forced on: with `auto` the copy would behave like the original on CUDA
        tc = cfg.transcription.model_copy(update={"model": model, "device": args.device, "single_encode": True})
        results, info, engines = {}, {}, {}
        for name, cls in (("original", WhisperEngine), ("single", SingleEncodeWhisperEngine)):
            eng = engines[name] = cls(tc)
            info[name] = eng.describe()
            results[name] = run(eng, utts, tc.language)
            if name == "single":
                info[name]["encoder_passes_per_utt"] = round(eng.encoder_passes / max(n_utts, 1), 2)
        o, s = results["original"], results["single"]
        same = flips = sampled = mismatch = 0
        details = []
        for cid in utts:
            for i, (a, b) in enumerate(zip(o[cid], s[cid])):
                if a["lang"] != b["lang"]:
                    flips += 1
                    details.append(f"    flip      {cid} utt {i}: {a['lang']} (p {a['p']}) -> {b['lang']} (p {b['p']})")
                elif a["text"] == b["text"]:
                    same += 1
                elif same_greedy(engines["original"], engines["single"], utts[cid][i].audio, tc.language):
                    sampled += 1                   # random temperature fallback, identical at T=0: not the change
                    details.append(f"    sampling  {cid} utt {i} ({a['lang']}): texts differ, identical at T=0")
                else:
                    mismatch += 1
                    details.append(f"    MISMATCH  {cid} utt {i} ({a['lang']}), also at T=0:\n"
                                   f"      original: {a['text']}\n      single:   {b['text']}")
        del engines
        t_o = sum(r["s"] for rows in o.values() for r in rows)
        t_s = sum(r["s"] for rows in s.values() for r in rows)
        dev = info["original"]["device"]
        line = (f"== {info['original']['model']} on {dev} ({info['original']['compute_type']}): identical "
                f"{same}/{n_utts} utterances, language flips {flips}, sampling-only differences {sampled}, "
                f"DECODE MISMATCHES {mismatch} | "
                f"RTF {t_o / audio_s:.2f} -> {t_s / audio_s:.2f} | per utterance {t_o / n_utts:.2f} -> "
                f"{t_s / n_utts:.2f} s ({(1 - t_s / t_o) * 100:.0f} % less) | encoder passes per utterance "
                f"{info['single']['encoder_passes_per_utt']}")
        refs = {cid: ref for cid, (_, ref) in data.items() if ref}
        if refs:
            w = {k: sum(wer(refs[c], " ".join(r["text"] for r in results[k][c] if r["text"])) for c in refs) / len(refs)
                 for k in ("original", "single")}
            line += f" | mean WER {w['original'] * 100:.1f}% -> {w['single'] * 100:.1f}%"
        print(line)
        print("\n".join(details) if details else "    (no differences)")
        failed |= mismatch > 0
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{info['original']['model']}-{dev}.json"
        path.write_text(json.dumps({"machine": platform.processor() or platform.machine(), "info": info,
                                    "summary": line, "results": results}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
