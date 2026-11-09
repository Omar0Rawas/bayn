"""Compare summary models on the realistic classroom scenarios (live transcripts with real errors).

    python -m bayn.summarization.evaluate                          # configured model, all variants, en + ar
    python -m bayn.summarization.evaluate --models qwen2.5:3b qwen2.5:7b llama3.2:3b --runs 2

Prints coverage / invented content / language / speed per model, and saves every summary
to data/eval/summaries-<date>.json for reading.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

from bayn.common.config import DATA_DIR, ROOT, load_config
from bayn.summarization.models import resolve_model
from bayn.summarization.ollama_client import OllamaClient, OllamaError, local_server
from bayn.summarization.prompts import build_messages, render, schema
from bayn.summarization.quality import check_summary

SCENARIOS = ROOT / "tests" / "data" / "scenarios" / "classroom.json"
LINE = re.compile(r"^\[([+-]\d+(?:\.\d+)?) s\] \((\w+)\) (.*)$")


def scenario_segments(lines: list[str], t_event: float = 1_800_000_000.0) -> list[dict]:
    """'[+012.8 s] (ar) text' lines -> transcript segments as the transcriber stores them."""
    out = []
    for i, line in enumerate(lines):
        m = LINE.match(line.strip())
        if m:
            t = t_event + float(m.group(1))
            out.append({"id": str(i), "t_start": t, "t_end": t + 2.0, "lang": m.group(2), "text": m.group(3)})
    return out


def _print_table(results: list[dict]) -> None:
    print(f"\n{'model':14s} {'output':>6s} {'pass':>6s} {'facts':>6s} {'invented':>9s} {'lang ok':>8s} {'avg s':>6s}")
    for model in dict.fromkeys(r["model"] for r in results):
        for lang in ("en", "ar", None):
            rs = [r for r in results if r["model"] == model and (lang is None or r["lang"] == lang)]
            if not rs:
                continue
            n = len(rs)
            print(f"{model:14s} {lang or 'all':>6s} {sum(r['quality']['passed'] for r in rs):3d}/{n:<2d} "
                  f"{sum(r['quality']['coverage'] for r in rs) / n * 100:5.0f}% "
                  f"{sum(bool(r['quality']['invented']) for r in rs):5d}/{n:<2d} "
                  f"{sum(r['quality']['language_ok'] for r in rs):5d}/{n:<2d} "
                  f"{sum(r['seconds'] for r in rs) / n:6.1f}")


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Score summary models on the classroom scenarios")
    ap.add_argument("--models", nargs="*", help="Ollama models (default: what `summary.model` resolves to)")
    ap.add_argument("--variants", nargs="*", default=["english", "arabic", "mixed"])
    ap.add_argument("--langs", nargs="*", default=["en", "ar"], help="summary output languages")
    ap.add_argument("--runs", type=int, default=1, help="repeat each case (models are not deterministic)")
    ap.add_argument("--rescore", help="re-check a saved data/eval/summaries-*.json with the current checks")
    ap.add_argument("--structured", action=argparse.BooleanOptionalAction, default=True,
                    help="JSON-constrained answers rendered by code (default) or free markdown (--no-structured)")
    args = ap.parse_args(argv)

    data = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    checks = data["summary_checks"]
    if args.rescore:
        results = json.loads(open(args.rescore, encoding="utf-8").read())
        for r in results:
            if not r.get("error"):
                r["quality"] = check_summary(r["summary"], r["lang"], checks, r["variant"]).to_dict()
        _print_table(results)
        return 0
    with local_server(cfg.summary.host) as ok:         # starts `ollama serve` for the run if needed
        if not ok:
            print(f"Ollama is not available at {cfg.summary.host}. Install it (bayn setup), then retry.")
            return 1
        return _evaluate(args, cfg, data, checks)


def _evaluate(args, cfg, data: dict, checks: dict) -> int:
    client = OllamaClient(cfg.summary.host, cfg.summary.timeout_s)
    event ={"t": 1_800_000_000.0, "detector": "manual", "source": "button", "label": "Manual trigger",
             "pre_s": 10, "post_s": 30}
    results = []
    for model in args.models or [resolve_model(cfg.summary, client.models())]:
        if not client.has_model(model):
            print(f"skip {model}: not pulled (ollama pull {model})")
            continue
        client.warm(model, cfg.summary.keep_alive)
        for variant in args.variants:
            segs = scenario_segments(data["variants"][variant]["live_transcript"])
            for lang in args.langs:
                for run in range(args.runs):
                    msgs = build_messages(event, segs, lang, cfg.summary.max_key_points, structured=args.structured)
                    try:
                        r = client.chat(model, msgs, cfg.summary.temperature, cfg.summary.keep_alive,
                                        json_schema=schema(cfg.summary.max_key_points) if args.structured else None)
                        if args.structured:
                            r.text = render(json.loads(r.text), lang)
                    except (OllamaError, ValueError) as e:   # stuck repeating itself / broken JSON
                        print(f"{model:14s} {variant:8s} -> {lang}  ERROR {e}", flush=True)
                        results.append({"model": model, "variant": variant, "lang": lang, "run": run,
                                        "seconds": 0.0, "tokens_per_s": 0, "error": str(e), "summary": "",
                                        "quality": {"passed": False, "coverage": 0.0, "invented": ["(error)"],
                                                    "language_ok": False}})
                        continue
                    q = check_summary(r.text, lang, checks, variant)
                    results.append({"model": model, "variant": variant, "lang": lang, "run": run,
                                    "seconds": round(r.total_s, 2), "tokens_per_s": round(r.tokens_per_s),
                                    "quality": q.to_dict(), "summary": r.text})
                    flag = "PASS" if q.passed else "fail"
                    n_facts = len(q.facts_found) + len(q.facts_missing)
                    print(f"{model:14s} {variant:8s} -> {lang}  {flag}  facts {len(q.facts_found)}/{n_facts}  "
                          f"invented {q.invented or '-'}  lang {'ok' if q.language_ok else 'WRONG'}  "
                          f"fmt {'ok' if q.format_ok else 'bad'}  {r.total_s:.1f}s", flush=True)
        # free GPU memory before the next model (several parked models starve Whisper + the live
        # summarizer - defect D17), unless it's the model the live service uses
        if model != resolve_model(cfg.summary, client.models()):
            client.unload(model)

    _print_table(results)
    out = DATA_DIR / "eval" / f"summaries-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAll summaries: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
