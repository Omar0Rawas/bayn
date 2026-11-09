"""Compare transcription models and strategies on recordings with a known script.

    python -m bayn.transcription.evaluate --corpus all                       # every recorded corpus script
    python -m bayn.transcription.evaluate --corpus all --models small large-v3-turbo --device cpu
    python -m bayn.transcription.evaluate --scenario mixed                    # the original classroom test
    python -m bayn.transcription.evaluate --wav my.wav --reference script.txt

The audio is segmented exactly like live (Silero VAD), then every model/strategy transcribes the
same utterances. Metrics: WER, CER, key-term recall (do the important terms survive?), real-time
factor. Results are saved to data/eval/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from bayn import corpus
from bayn.common.config import DATA_DIR, ROOT, load_config
from bayn.transcription.audio_sources import read_wav
from bayn.transcription.engine import WhisperEngine, silero_vad
from bayn.transcription.metrics import cer, key_term_recall, script_mix, wer
from bayn.transcription.segmenter import Segmenter, Utterance

SCENARIOS = ROOT / "tests" / "data" / "scenarios" / "classroom.json"

# name -> (language mode, mixed strategy, initial prompt)
STRATEGIES = {
    "mixed (current)": ("mixed", "per_utterance", "none"),
    "mixed + arabic_lean": ("mixed", "arabic_lean", "none"),
    "mixed + prompt": ("mixed", "per_utterance", "auto"),
    "mixed + arabic_lean + prompt": ("mixed", "arabic_lean", "auto"),
    "auto (free detection)": ("auto", None, "none"),
    "forced ar": ("ar", None, "none"),
    "forced ar + prompt": ("ar", None, "auto"),
    "forced en": ("en", None, "none"),
}


def segment(audio, min_silence_s: float) -> list[Utterance]:
    seg = Segmenter(silero_vad(), min_silence_s=min_silence_s)
    out = []
    for i in range(0, len(audio), 1600):
        seg.feed(audio[i:i + 1600], i / 16000)
        out += seg.step().finals
    return out + seg.flush()


def evaluate(engine: WhisperEngine, audio, reference: str, strategies: dict, min_silence_s: float = 0.6,
             key_terms: list[str] | None = None) -> list[dict]:
    utts = segment(audio, min_silence_s)
    audio_s = len(audio) / 16000
    rows = []
    for name, (mode, strategy, prompt) in strategies.items():
        parts, langs, secs = [], [], 0.0
        for u in utts:
            r = engine.transcribe(u.audio, mode, strategy=strategy, prompt=prompt)
            parts.append(r.text)
            langs.append(r.language)
            secs += r.seconds
        text = " ".join(p for p in parts if p)
        mix = script_mix(text)
        recall, missing = key_term_recall(key_terms or [], text)
        rows.append({"strategy": name, "wer": wer(reference, text), "cer": cer(reference, text),
                     "key_term_recall": recall, "missing_terms": missing,
                     "latin_share": round(mix["latin"], 3), "utterances": len(utts), "languages": langs,
                     "seconds": round(secs, 2), "rtf": round(secs / audio_s, 3) if audio_s else None, "text": text})
    return rows


# default strategies per corpus version (all of them with --strategies)
CORPUS_STRATEGIES = {
    "en": ["mixed (current)", "forced en"],
    "ar": ["mixed (current)", "forced ar", "forced ar + prompt"],
    "mixed": ["mixed (current)", "mixed + arabic_lean", "mixed + prompt", "mixed + arabic_lean + prompt",
              "forced ar"],
}


def run_corpus(args, cfg) -> int:
    """Benchmark: every recorded corpus script x model x strategy, on identical audio."""
    ids = [s["id"] for s in corpus.scripts()] if args.corpus == "all" else [args.corpus]
    todo = [(corpus.get(i), corpus.recording_path(i)) for i in ids]
    missing = [s["id"] for s, p in todo if s and not p.exists()]
    todo = [(s, p) for s, p in todo if s and p.exists()]
    if missing:
        print("Not recorded yet (skipped): " + ", ".join(missing))
    if not todo:
        print("Nothing to evaluate. Record first, e.g.:  .\\bayn record mx-pi")
        return 1
    models = args.models or [cfg.transcription.model]
    results = []
    for model in models:
        tc = cfg.transcription.model_copy(update={"model": model, **({"device": args.device} if args.device else {})})
        try:
            engine = WhisperEngine(tc)
        except RuntimeError as e:
            print(f"skip model {model}: {e}")
            continue
        d = engine.describe()
        print(f"\n== Whisper {d['model']} on {engine.device} ({d['compute_type']}) ==")
        for script, wav in todo:
            names = args.strategies or CORPUS_STRATEGIES[script["language"]]
            rows = evaluate(engine, read_wav(wav), script["text"], {n: STRATEGIES[n] for n in names},
                            cfg.transcription.min_silence_s, script["key_terms"])
            for r in rows:
                r.update({"script": script["id"], "language": script["language"], "model": d["model"],
                          "device": engine.device})
                results.append(r)
                print(f"  {script['id']:12s} {r['strategy']:30s} WER {r['wer'] * 100:5.1f}%  CER {r['cer'] * 100:5.1f}%  "
                      f"terms {r['key_term_recall'] * 100:4.0f}%  RTF {r['rtf']:.2f}"
                      + (f"  missing: {', '.join(r['missing_terms'])}" if r["missing_terms"] else ""), flush=True)
    # summary per model x strategy x language version
    print(f"\n{'model':16s} {'device':6s} {'version':7s} {'strategy':30s} {'WER':>6s} {'CER':>6s} {'terms':>6s} {'RTF':>5s}")
    keyf = lambda r: (r["model"], r["device"], r["language"], r["strategy"])
    for key in dict.fromkeys(keyf(r) for r in results):
        rs = [r for r in results if keyf(r) == key]
        n = len(rs)
        print(f"{key[0]:16s} {key[1]:6s} {key[2]:7s} {key[3]:30s} {sum(r['wer'] for r in rs) / n * 100:5.1f}% "
              f"{sum(r['cer'] for r in rs) / n * 100:5.1f}% {sum(r['key_term_recall'] for r in rs) / n * 100:5.0f}% "
              f"{sum(r['rtf'] for r in rs) / n:5.2f}")
    out = DATA_DIR / "eval" / f"stt-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAll transcripts: {out}\n(RTF = processing time / audio length; below 1 = faster than real time)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score transcription strategies against a script")
    ap.add_argument("--scenario", choices=["english", "arabic", "mixed"],
                    help="use the classroom script + data/recordings/classroom_<scenario>.wav")
    ap.add_argument("--wav")
    ap.add_argument("--reference", help="reference text, or a path to a UTF-8 text file")
    ap.add_argument("--strategies", nargs="*", help="subset of strategy names (default: all relevant)")
    ap.add_argument("--corpus", help="a corpus script id (e.g. mx-pi) or 'all': uses data/recordings/<id>.wav")
    ap.add_argument("--models", nargs="*", help="Whisper models to compare, e.g. tiny small large-v3-turbo")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], help="force GPU or CPU (e.g. to mimic a laptop)")
    args = ap.parse_args(argv)

    if args.corpus:
        return run_corpus(args, load_config())
    if args.scenario:
        sc = json.loads(SCENARIOS.read_text(encoding="utf-8"))["variants"][args.scenario]
        reference = sc["reference"]
        wav = Path(args.wav or DATA_DIR / "recordings" / f"classroom_{args.scenario}.wav")
    else:
        if not (args.wav and args.reference):
            ap.error("give --scenario, or --wav and --reference")
        ref = Path(args.reference)
        reference = ref.read_text(encoding="utf-8") if ref.exists() else args.reference
        wav = Path(args.wav)
    if not wav.exists():
        print(f"Recording not found: {wav}\nRecord it first:  python -m bayn.transcription.record {wav.stem}")
        return 1

    names = args.strategies or [n for n in STRATEGIES
                                if not (args.scenario == "english" and n.startswith("forced ar"))]
    strategies = {n: STRATEGIES[n] for n in names}
    cfg = load_config()
    engine = WhisperEngine(cfg.transcription)
    print(f"Engine: {engine.describe()['model']} on {engine.device}.  Recording: {wav.name}")
    rows = evaluate(engine, read_wav(wav), reference, strategies, cfg.transcription.min_silence_s)

    rows.sort(key=lambda r: r["wer"])
    print(f"\n{'strategy':32s} {'WER':>7s} {'CER':>7s} {'Latin':>6s} {'time':>6s}")
    for r in rows:
        print(f"{r['strategy']:32s} {r['wer'] * 100:6.1f}% {r['cer'] * 100:6.1f}% "
              f"{r['latin_share'] * 100:5.0f}% {r['seconds']:5.1f}s")
    print(f"\nBest: {rows[0]['strategy']}\n{rows[0]['text']}")
    out = wav.with_suffix(".eval.json")
    out.write_text(json.dumps({"wav": wav.name, "reference": reference, "engine": engine.describe(), "results": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetails: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
