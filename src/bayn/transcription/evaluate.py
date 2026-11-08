"""Compare transcription strategies on a recording with a known script (WER / CER).

    python -m bayn.transcription.evaluate --scenario mixed          # data/recordings/classroom_mixed.wav
    python -m bayn.transcription.evaluate --wav my.wav --reference script.txt --mode mixed

The audio is segmented exactly like live (Silero VAD), then every strategy transcribes the
same utterances. Results print as a table and are saved next to the WAV (<name>.eval.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bayn.common.config import DATA_DIR, ROOT, load_config
from bayn.transcription.audio_sources import read_wav
from bayn.transcription.engine import WhisperEngine, silero_vad
from bayn.transcription.metrics import cer, script_mix, wer
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


def evaluate(engine: WhisperEngine, audio, reference: str, strategies: dict, min_silence_s: float = 0.6) -> list[dict]:
    utts = segment(audio, min_silence_s)
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
        rows.append({"strategy": name, "wer": wer(reference, text), "cer": cer(reference, text),
                     "latin_share": round(mix["latin"], 3), "utterances": len(utts),
                     "languages": langs, "seconds": round(secs, 2), "text": text})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score transcription strategies against a script")
    ap.add_argument("--scenario", choices=["english", "arabic", "mixed"],
                    help="use the classroom script + data/recordings/classroom_<scenario>.wav")
    ap.add_argument("--wav")
    ap.add_argument("--reference", help="reference text, or a path to a UTF-8 text file")
    ap.add_argument("--strategies", nargs="*", help="subset of strategy names (default: all relevant)")
    args = ap.parse_args(argv)

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
