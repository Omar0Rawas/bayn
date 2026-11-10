"""Benchmark Whisper x summary-LLM combinations on the recorded corpus (docs/CORPUS.md).

    python -m bayn.benchmark stt --models tiny small large-v3-turbo --profiles gpu laptop
    python -m bayn.benchmark llm --models qwen2.5:3b qwen2.5:7b --sources reference        # LLM alone
    python -m bayn.benchmark llm --models qwen2.5:3b --sources all                         # every Whisper output
    python -m bayn.benchmark llm --models qwen2.5:3b --profile laptop --scripts en-incident ar-incident mx-incident
    python -m bayn.benchmark report                                                         # -> docs/BENCHMARK.md

Profiles (the machines we care about):
  gpu     NVIDIA GPU as found (Whisper float16, beam 5; Ollama on the GPU)
  laptop  CPU only, 4 threads (Whisper int8, beam 1; Ollama with num_gpu=0) - a modest laptop without a
          usable GPU, emulated on the dev PC. A desktop core is faster than a laptop core: best case.
  cpu     CPU only with all of this machine's cores - use it ON the demo laptop for the real numbers:
            python -m bayn.benchmark stt --models tiny base small medium --profiles cpu
            python -m bayn.benchmark llm --profile cpu --sources all --models gemma4:e2b qwen3:4b
            python -m bayn.benchmark mem --profiles cpu --models gemma4:e2b qwen3:4b
            python -m bayn.benchmark report

Every step is cached in data/eval/bench/ (gitignored: transcripts of the user's voice), so a run can be
stopped and resumed; --force redoes it. Each Whisper model runs in its own process so its memory use is
measured cleanly. Summary *quality* doesn't depend on the device (same model file), so quality runs use
the GPU and the laptop profile is only needed for speed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from bayn import corpus
from bayn.common.config import DATA_DIR, ROOT, load_config

BENCH = DATA_DIR / "eval" / "bench"
STT_DIR, LLM_DIR = BENCH / "stt", BENCH / "llm"
REPORT = ROOT / "docs" / "BENCHMARK.md"
T0 = 1_800_000_000.0
EVENT = {"t": T0, "detector": "manual", "source": "button", "label": "Manual trigger", "pre_s": 10, "post_s": 30}
VERSIONS = ("en", "ar", "mixed")
PROFILES = {
    "gpu": {"label": "NVIDIA GPU", "device": "cuda", "threads": "auto", "llm": {}},
    "laptop": {"label": "CPU only, 4 threads", "device": "cpu", "threads": 4, "llm": {"num_gpu": 0, "num_thread": 4}},
    "cpu": {"label": "CPU only, this machine's cores", "device": "cpu", "threads": "auto", "llm": {"num_gpu": 0}},
}
CPU_PROFILES = ("cpu", "laptop")   # run on the real laptop: --profiles cpu (+ gpu if it has an NVIDIA GPU)
# Hardware tiers for the recommendation: memory budget for Whisper + LLM together (headroom left for the
# dashboard / CUDA context), and for CPU-only machines the speed needed to keep up live.
GPU_TIERS = [("NVIDIA GPU with 10-12 GB", 9500), ("NVIDIA GPU with 8 GB", 7000), ("NVIDIA GPU with 6 GB", 5000),
             ("NVIDIA GPU with 4 GB", 3300)]
CPU_TIERS = [("CPU only, 16 GB RAM", 9000), ("CPU only, 8 GB RAM", 4500)]
MAX_CPU_RTF = 0.5          # Whisper must use at most half of real time on the CPU (EEG + UI run too)
MAX_CPU_SUMMARY_S = 60     # a summary should be on screen within a minute of the window closing
TIE = 0.03                 # summary-score differences below this are noise (one run per case)
CER_TIE = 0.02             # transcripts within 2 points CER count as equally good -> prefer the lighter combo


def safe(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")


def recorded() -> list[dict]:
    return [s for s in corpus.scripts() if corpus.recording_path(s["id"]).exists()]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- measurements

def peak_rss_mb() -> float:
    """Peak resident memory of this process (MB)."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                       [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize",
                                                       "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                                                       "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                                       "PagefileUsage", "PeakPagefileUsage")]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2**20
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


def gpu_used_mb() -> int | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def ollama_ram_mb() -> int | None:
    """Resident memory of all Ollama processes (server + model runners such as llama-server), MB."""
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=10).stdout
            return sum(int("".join(c for c in line.rsplit('","', 1)[-1] if c.isdigit())) for line in out.splitlines()
                       if "llama" in line.split('","', 1)[0].lower()) // 1024
        out = subprocess.run(["ps", "-eo", "rss=,comm="], capture_output=True, text=True, timeout=10).stdout
        return sum(int(l.split()[0]) for l in out.splitlines() if "llama" in l.lower()) // 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def machine() -> dict:
    gpu = ""
    if shutil.which("nvidia-smi"):
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    cpu = platform.processor()
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return {"cpu": cpu, "cpu_count": os.cpu_count(), "gpu": gpu, "os": platform.platform(terse=True)}


# ---------------------------------------------------------------- Whisper

def stt_path(profile: str, model: str) -> Path:
    return STT_DIR / profile / f"{safe(model)}.json"


def stt_worker(model: str, profile: str) -> int:
    """One Whisper model on one profile, in a fresh process (clean memory numbers)."""
    from bayn.transcription.audio_sources import read_wav
    from bayn.transcription.engine import WhisperEngine
    from bayn.transcription.evaluate import segment
    from bayn.transcription.metrics import cer, key_term_recall, wer

    p, cfg = PROFILES[profile], load_config()
    tc = cfg.transcription.model_copy(update={"model": model, "device": p["device"], "cpu_threads": p["threads"]})
    scripts = recorded()
    # VAD segmentation first (the same for every model), so it is part of the memory baseline
    audio = {s["id"]: read_wav(corpus.recording_path(s["id"])) for s in scripts}
    utts = {sid: segment(a, tc.min_silence_s) for sid, a in audio.items()}
    base_mb, vram0 = peak_rss_mb(), gpu_used_mb()
    try:
        engine = WhisperEngine(tc)
    except RuntimeError as e:
        print(f"  {model}@{profile}: {e}")
        return 2
    if engine.device != p["device"]:
        print(f"  {model}@{profile}: fell back to {engine.device} ({engine.fallback_reason}) - not measured")
        return 2
    out = {"model": model, "profile": profile, "engine": engine.describe(), "machine": machine(),
           "language_mode": tc.language, "scripts": {}}
    for s in scripts:
        segs, secs = [], 0.0
        for i, u in enumerate(utts[s["id"]]):
            r = engine.transcribe(u.audio, tc.language)
            secs += r.seconds
            if r.text:
                segs.append({"id": str(i), "t_start": T0 + u.t_start, "t_end": T0 + u.t_start + len(u.audio) / u.sr,
                             "lang": r.language, "text": r.text})
        text = " ".join(x["text"] for x in segs)
        audio_s = len(audio[s["id"]]) / 16000
        recall, missing = key_term_recall(s["key_terms"], text)
        out["scripts"][s["id"]] = {
            "version": s["language"], "wer": round(wer(s["text"], text), 4), "cer": round(cer(s["text"], text), 4),
            "key_term_recall": round(recall, 3), "missing_terms": missing, "seconds": round(secs, 2),
            "audio_s": round(audio_s, 1), "rtf": round(secs / audio_s, 3), "segments": segs, "text": text}
        print(f"  {model:15s} {profile:6s} {s['id']:12s} WER {out['scripts'][s['id']]['wer'] * 100:5.1f}%  "
              f"terms {recall * 100:4.0f}%  RTF {secs / audio_s:.2f}"
              + (f"  missing: {', '.join(missing)}" if missing else ""), flush=True)
    vram1 = gpu_used_mb()
    out["ram_mb"] = round(peak_rss_mb() - base_mb)
    out["vram_mb"] = (vram1 - vram0) if profile == "gpu" and vram0 is not None and vram1 is not None else None
    _save(stt_path(profile, model), out)
    return 0


def run_stt(args) -> int:
    ids = {s["id"] for s in recorded()}
    if not ids:
        print("No corpus recordings yet: .\\bayn record <id>  (docs/CORPUS.md)")
        return 1
    for profile in args.profiles:
        for model in args.models:
            done = _load(stt_path(profile, model))
            if done and set(done.get("scripts", {})) >= ids and not args.force:
                print(f"  {model}@{profile}: cached")
                continue
            print(f"== Whisper {model} on {profile} ({PROFILES[profile]['label']}) ==", flush=True)
            subprocess.run([sys.executable, "-m", "bayn.benchmark", "_stt", model, profile], check=False)
    return 0


# ---------------------------------------------------------------- summaries

def stt_sources() -> list[str]:
    return [f"{p.parent.name}/{p.stem}" for p in sorted(STT_DIR.glob("*/*.json"))]


def source_segments(source: str) -> dict[str, list[dict]]:
    """script id -> transcript segments. 'reference' = the script itself (a perfect transcriber)."""
    if source == "reference":
        out = {}
        for s in recorded() or corpus.scripts():
            lang = "en" if s["language"] == "en" else "ar"
            out[s["id"]] = [{"id": str(i), "t_start": T0 + 3 * i, "t_end": T0 + 3 * i + 3, "lang": lang, "text": t}
                            for i, t in enumerate(corpus.sentences(s["text"]))]
        return out
    profile, model = source.split("/", 1)
    data = _load(STT_DIR / profile / f"{model}.json")
    return {sid: v["segments"] for sid, v in data.get("scripts", {}).items()}


def llm_path(profile: str, model: str, source: str) -> Path:
    return LLM_DIR / profile / safe(model) / f"{safe(source)}.json"


def mem_path(profile: str, model: str) -> Path:
    return BENCH / "mem" / profile / f"{safe(model)}.json"


def run_mem(args) -> int:
    """Memory a summary model really uses, measured from outside (GPU driver / process RAM) while it
    writes one summary. Ollama's own /api/ps under-reports some models (Gemma 4: 0.2 of 4.3 GB)."""
    from bayn.summarization.ollama_client import OllamaClient, local_server
    from bayn.summarization.prompts import build_messages, schema

    cfg = load_config()
    script = corpus.get("mx-incident")
    msgs = build_messages(EVENT, source_segments("reference")[script["id"]], "ar", cfg.summary.max_key_points)
    with local_server(cfg.summary.host) as ok:
        if not ok:
            return 1
        client = OllamaClient(cfg.summary.host, timeout_s=600, stall_s=180)
        for profile in args.profiles:
            opts = PROFILES[profile]["llm"]
            for model in args.models:
                if mem_path(profile, model).exists() and not args.force:
                    continue
                for other in client.loaded():
                    client.unload(other["name"])
                time.sleep(2)
                vram0, ram0 = gpu_used_mb(), ollama_ram_mb()
                client.chat(model, msgs, cfg.summary.temperature, "1m", json_schema=schema(cfg.summary.max_key_points),
                            options=opts)
                vram1, ram1 = gpu_used_mb(), ollama_ram_mb()
                res = {"model": model, "profile": profile,
                       "vram_mb": (vram1 - vram0) if None not in (vram0, vram1) else None,
                       "ram_mb": (ram1 - ram0) if None not in (ram0, ram1) else None}
                client.unload(model)
                _save(mem_path(profile, model), res)
                print(f"  {model:14s} {profile:6s} GPU +{res['vram_mb']} MB  RAM +{res['ram_mb']} MB", flush=True)
    return 0


def run_llm(args) -> int:
    from bayn.summarization.ollama_client import OllamaClient, OllamaError, local_server
    from bayn.summarization.prompts import build_messages, render, schema
    from bayn.summarization.quality import check_summary

    cfg = load_config()
    sources = []
    for s in args.sources:
        sources += (["reference"] + stt_sources()) if s == "all" else [s]
    opts = PROFILES[args.profile]["llm"]
    with local_server(cfg.summary.host) as ok:
        if not ok:
            print(f"Ollama is not available at {cfg.summary.host}")
            return 1
        client = OllamaClient(cfg.summary.host, timeout_s=600, stall_s=180 if args.profile in CPU_PROFILES else 60)
        for model in args.models:
            if not client.has_model(model):
                print(f"skip {model}: not pulled (python -m bayn.summarization.models {model})")
                continue
            for other in client.loaded():                    # one model at a time: clean memory + speed numbers
                client.unload(other["name"])
            try:
                load_s = client.warm(model, "10m", opts)
            except (OllamaError, OSError) as e:
                print(f"skip {model}: {e}")
                continue
            mem = next((m for m in client.loaded() if m["name"].startswith(model)), {})
            memory = {"size_mb": round(mem.get("size", 0) / 2**20), "vram_mb": round(mem.get("size_vram", 0) / 2**20),
                      "load_s": round(load_s, 1)}
            for source in sources:
                transcripts = source_segments(source)
                path = llm_path(args.profile, model, source)
                data = _load(path) or {"model": model, "source": source, "profile": args.profile, "cases": {}}
                data["memory"], data["machine"] = memory, machine()
                for s in recorded():
                    if args.scripts and s["id"] not in args.scripts or s["id"] not in transcripts:
                        continue
                    for lang in args.langs:
                        key = f"{s['id']}/{lang}"
                        if key in data["cases"] and not args.force and not data["cases"][key].get("error"):
                            continue
                        msgs = build_messages(EVENT, transcripts[s["id"]], lang, cfg.summary.max_key_points)
                        case = {"script": s["id"], "version": s["language"], "lang": lang}
                        try:
                            r = client.chat(model, msgs, cfg.summary.temperature, "10m",
                                            json_schema=schema(cfg.summary.max_key_points), options=opts)
                            text = render(json.loads(r.text), lang)
                            q = check_summary(text, lang, corpus.checks_for(s))
                            case.update(seconds=round(r.total_s, 2), first_token_s=round(r.first_token_s, 2),
                                        tokens=r.tokens, tokens_per_s=round(r.tokens_per_s, 1),
                                        quality=q.to_dict(), summary=text)
                            msg = (f"{'PASS' if q.passed else 'fail'} facts {q.coverage * 100:3.0f}%  "
                                   f"invented {q.invented or '-'}  {r.total_s:5.1f}s")
                        except (OllamaError, ValueError) as e:
                            case.update(error=str(e))
                            msg = f"ERROR {e}"
                        data["cases"][key] = case
                        _save(path, data)
                        print(f"  {model:14s} {args.profile:6s} {source:24s} {key:16s} {msg}", flush=True)
            client.unload(model)
    return 0


# ---------------------------------------------------------------- report

def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _pct(x: float | None, digits: int = 0) -> str:
    return "–" if x is None else f"{x * 100:.{digits}f} %"


def _cases(data: dict) -> list[dict]:
    return [c for c in data.get("cases", {}).values()]


def quality_stats(cases: list[dict]) -> dict:
    ok = [c for c in cases if not c.get("error")]
    n = len(cases)
    return {"n": n, "errors": n - len(ok),
            "facts": _mean([c["quality"]["coverage"] for c in ok] + [0.0] * (n - len(ok))) if n else None,
            "invented": sum(bool(c["quality"]["invented"]) for c in ok),
            "wrong_lang": sum(not c["quality"]["language_ok"] for c in ok),
            "passed": sum(c["quality"]["passed"] for c in ok),
            "seconds": _mean([c["seconds"] for c in ok])}


def load_results() -> tuple[dict, dict]:
    """Cached results, re-scored with the current metrics / checks (no need to re-run models)."""
    from bayn.summarization.quality import check_summary
    from bayn.transcription.metrics import cer, key_term_recall, wer

    stt = {}
    for p in sorted(STT_DIR.glob("*/*.json")):
        d = _load(p)
        for sid, x in d.get("scripts", {}).items():
            if s := corpus.get(sid):
                recall, missing = key_term_recall(s["key_terms"], x["text"])
                x.update(wer=wer(s["text"], x["text"]), cer=cer(s["text"], x["text"]),
                         key_term_recall=recall, missing_terms=missing)
        stt[f"{p.parent.name}/{p.stem}"] = d
    llm = {}
    for p in sorted(LLM_DIR.glob("*/*/*.json")):
        d = _load(p)
        for c in d.get("cases", {}).values():
            if not c.get("error") and (s := corpus.get(c["script"])):
                c["quality"] = check_summary(c["summary"], c["lang"], corpus.checks_for(s)).to_dict()
        # measured memory (bench mem) beats Ollama's own report; the GPU number also applies to the
        # quality runs, the RAM number to CPU-only use
        mem = d.setdefault("memory", {})
        if (g := _load(mem_path("gpu", d["model"]))).get("vram_mb"):
            mem["vram_mb"] = g["vram_mb"]
        for prof in CPU_PROFILES:
            if (cpu := _load(mem_path(prof, d["model"]))).get("ram_mb"):
                mem["size_mb"] = cpu["ram_mb"]
                break
        llm[(d["profile"], d["model"], d["source"])] = d
    return stt, llm


def quality_profile(llm: dict) -> str:
    """Profile whose LLM runs measure summary quality: `gpu` on a machine with one (dev PC; CPU profiles there
    only measure speed), else the CPU profile that ran (demo laptop without an NVIDIA GPU)."""
    profiles = {p for (p, _, _) in llm}
    return "gpu" if "gpu" in profiles else next((p for p in CPU_PROFILES if p in profiles), "gpu")


def cpu_speed(llm: dict) -> dict:
    """(profile, model) -> quality_stats over ALL its CPU runs (every transcript source) + measured RAM."""
    out = {}
    for (profile, model, _), d in llm.items():
        if profile in CPU_PROFILES:
            e = out.setdefault((profile, model), {"cases": [], "size_mb": 0})
            e["cases"] += _cases(d)
            e["size_mb"] = d.get("memory", {}).get("size_mb", 0) or e["size_mb"]
    return {k: quality_stats(v["cases"]) | {"size_mb": v["size_mb"], "ok": [c for c in v["cases"] if not c.get("error")]}
            for k, v in out.items()}


def recommend(stt: dict, llm: dict) -> list[tuple[str, str, str, str]]:
    """Best measured combination per hardware tier: (tier, whisper, llm, why)."""
    rows = []
    qp = quality_profile(llm)
    combos = []      # (source, model, stats) from the quality runs (GPU on the dev PC, CPU on the laptop)
    for (profile, model, source), d in llm.items():
        if profile == qp and source != "reference" and source in stt:
            st = quality_stats(_cases(d))
            if st["n"]:
                combos.append((source, model, st, d.get("memory", {})))
    rank = lambda c: (c[2]["facts"] or 0) - 0.05 * c[2]["invented"] - 0.05 * c[2]["wrong_lang"]

    def cer_of(source: str) -> float:
        return _mean([x["cer"] for x in stt[source]["scripts"].values()]) or 1.0

    def pick(cands, key, source, mem):
        """Best summary score. Within TIE of it (one run per case: smaller differences are noise) prefer the
        better live transcript (the teacher reads it too), and among those the lightest combination."""
        top = max(key(c) for c in cands)
        near = [c for c in cands if key(c) >= top - TIE]
        best = min(cer_of(source(c)) for c in near)
        return min((c for c in near if cer_of(source(c)) <= best + CER_TIE), key=mem)
    for tier, budget in GPU_TIERS:
        fits = [c for c in combos if c[0].startswith("gpu/") and stt[c[0]].get("vram_mb") is not None
                and stt[c[0]]["vram_mb"] + c[3].get("vram_mb", 0) <= budget and c[3].get("vram_mb", 0) > 0]
        if fits:
            b = pick(fits, rank, lambda c: c[0], lambda c: stt[c[0]]["vram_mb"] + c[3]["vram_mb"])
            mem = stt[b[0]]["vram_mb"] + b[3]["vram_mb"]
            rows.append((tier, b[0].split("/")[1], b[1], f"facts {_pct(b[2]['facts'])}, ~{mem / 1024:.1f} GB VRAM"))
        else:
            rows.append((tier, "–", "–", "no measured combination fits"))
    cpu_llm = cpu_speed(llm)
    for tier, budget in CPU_TIERS:
        fits = []
        for c in combos:
            profile = c[0].split("/")[0]                 # CPU transcripts + LLM speed on the same profile
            if profile not in CPU_PROFILES or (profile, c[1]) not in cpu_llm:
                continue
            w, sp = stt[c[0]], cpu_llm[(profile, c[1])]
            rtf = _mean([v["rtf"] for v in w["scripts"].values()])
            if rtf is not None and rtf <= MAX_CPU_RTF and (sp["seconds"] or 1e9) <= MAX_CPU_SUMMARY_S \
                    and w.get("ram_mb", 0) + sp["size_mb"] <= budget:
                fits.append((c, rtf, sp))
        if fits:
            (src, model, st, _), rtf, sp = pick(fits, lambda f: rank(f[0]), lambda f: f[0][0],
                                                lambda f: stt[f[0][0]].get("ram_mb", 0) + f[2]["size_mb"])
            rows.append((tier, src.split("/")[1], model, f"facts {_pct(st['facts'])}, Whisper RTF {rtf:.2f}, "
                                                         f"summary ~{sp['seconds']:.0f} s ({PROFILES[src.split('/')[0]]['label']})"))
        else:
            rows.append((tier, "–", "–", f"nothing measured is fast enough (RTF ≤ {MAX_CPU_RTF}, summary ≤ "
                                         f"{MAX_CPU_SUMMARY_S} s) and fits"))
    return rows


def render_report() -> str:
    stt, llm = load_results()
    qp = quality_profile(llm)
    n_scripts = len(recorded())
    mach = next((d["machine"] for d in stt.values() if d.get("machine")), machine())
    where = ("> **Current results are from the dev PC** (the `laptop` profile only emulates a modest laptop with 4 "
             "CPU threads). **The best demo configuration will be re-tested on the demo laptop itself**, with its "
             "own microphone and the `cpu` profile, before the hackathon." if qp == "gpu" else
             f"> **Results from this machine without a GPU** (`{qp}` profile: Whisper and the LLM both on the CPU, "
             "measured one at a time; in the live app they share it). Summary quality here comes from CPU runs.")
    out = ["# Bayn model benchmark", "",
           "Which Whisper model and which summary LLM to use, per language and per machine. Measured on the "
           f"user's own recordings of the [corpus](CORPUS.md) ({n_scripts} scripts: English, Arabic, and mixed "
           "Arabic with English key terms), with the same settings as the live demo.", "",
           "> Generated by `python -m bayn.benchmark report` from data/eval/bench/ (raw transcripts and summaries "
           "stay on the machine that ran them).", "", where, "",
           f"**Machine:** {mach.get('cpu')} ({mach.get('cpu_count')} threads), {mach.get('gpu') or 'no NVIDIA GPU'}. "
           f"**Profiles:** `gpu` = Whisper float16 + beam 5 on the GPU, LLM on the GPU; `laptop` = CPU only with "
           f"4 threads (Whisper int8 + beam 1, LLM with num_gpu=0) to mimic a modest laptop. Desktop cores are "
           f"faster than laptop cores, so laptop speeds are a best case; `cpu` = CPU only with all cores (for "
           f"running this benchmark on the demo laptop itself - see `python -m bayn.benchmark --help`). Each "
           f"case ran once: differences of a few points are noise.", "",
           "**Metrics:** WER = word error rate, CER = character error rate (fairer for Arabic, whose spelling "
           "varies), key terms = share of each script's must-keep terms that survive (Rutherford, pi, chlorophyll, "
           "...), RTF = processing time / audio length (below 1 = faster than real time). Summaries: facts = share "
           "of the topic's key facts present; invented = summaries with forbidden content (diagnoses, unsupported "
           "claims); a summary passes with ≥ 80 % facts, nothing invented, the right language and all sections.", ""]

    # --- Whisper
    out += ["## 1. Transcription (Whisper)", "",
            "| Model | Profile | WER en | WER ar | WER mixed | CER ar | CER mixed | Key terms en / ar / mixed | RTF | "
            "Load | Memory |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, d in stt.items():
        sc = d.get("scripts", {}).values()
        by = {v: [x for x in sc if x["version"] == v] for v in VERSIONS}
        m = lambda v, f: _mean([x[f] for x in by[v]])
        mem = f"{d['vram_mb'] / 1024:.1f} GB VRAM" if d.get("vram_mb") else f"{d.get('ram_mb', 0) / 1024:.1f} GB RAM"
        out.append(f"| {d['model']} | {d['profile']} | {_pct(m('en', 'wer'), 1)} | {_pct(m('ar', 'wer'), 1)} | "
                   f"{_pct(m('mixed', 'wer'), 1)} | {_pct(m('ar', 'cer'), 1)} | {_pct(m('mixed', 'cer'), 1)} | "
                   f"{_pct(m('en', 'key_term_recall'))} / {_pct(m('ar', 'key_term_recall'))} / "
                   f"{_pct(m('mixed', 'key_term_recall'))} | {_mean([x['rtf'] for x in sc]):.2f} | "
                   f"{d['engine'].get('load_s', 0):.0f} s | {mem} |")
    lost = {}
    for d in stt.values():
        for sid, x in d.get("scripts", {}).items():
            for t in x["missing_terms"]:
                lost.setdefault(f"{d['model']}@{d['profile']}", []).append(f"{t} ({sid})")
    if lost:
        out += ["", "<details><summary>Key terms lost, per model</summary>", ""]
        out += [f"- **{k}:** {', '.join(v)}" for k, v in lost.items()]
        out += ["", "</details>"]

    # --- LLM alone
    ref = {(p, m): d for (p, m, s), d in llm.items() if s == "reference"}
    out += ["", "## 2. Summaries from a perfect transcript (the LLM alone)", "",
            "Input = the script text, so only the LLM is measured. Each model summarises every script in English "
            "and in Arabic.", "",
            "| LLM | Facts (all) | from en / ar / mixed speech | written in en / ar | Invented | Wrong language | "
            f"Pass | Time ({'GPU' if qp == 'gpu' else qp})"
            f" | {'GPU memory' if qp == 'gpu' else 'RAM'} |", "|---|---|---|---|---|---|---|---|---|"]
    for (p, m), d in sorted(ref.items(), key=lambda kv: -(quality_stats(_cases(kv[1]))["facts"] or 0)):
        if p != qp:
            continue
        cs = _cases(d)
        st = quality_stats(cs)
        f = lambda pred: _pct(quality_stats([c for c in cs if pred(c)])["facts"])
        out.append(f"| {m} | **{_pct(st['facts'])}** | {f(lambda c: c['version'] == 'en')} / "
                   f"{f(lambda c: c['version'] == 'ar')} / {f(lambda c: c['version'] == 'mixed')} | "
                   f"{f(lambda c: c['lang'] == 'en')} / {f(lambda c: c['lang'] == 'ar')} | {st['invented']}/{st['n']} | "
                   f"{st['wrong_lang']}/{st['n']} | {st['passed']}/{st['n']} | {st['seconds'] or 0:.1f} s | "
                   f"{d.get('memory', {}).get('vram_mb' if qp == 'gpu' else 'size_mb', 0) / 1024:.1f} GB |")

    # --- combinations
    models = sorted({m for (p, m, s) in llm if p == qp and s != "reference"})
    sources = [s for s in stt if any(((qp, m, s) in llm) for m in models)]
    if models and sources:
        out += ["", "## 3. Whisper × LLM combinations (real transcripts of your voice)", "",
                "Facts in the summary when the LLM gets the transcript that Whisper actually produced "
                "(summaries in English and Arabic). Rows marked `laptop` / `cpu` use the CPU int8 transcripts. "
                "Arabic summaries may keep English terms in Arabic letters (النيوكليوس = nucleus), as Whisper "
                "writes them; that counts, because a bilingual reader understands it.", ""]
        for title, pred in [("All scripts", lambda c: True)] + \
                           [(f"{corpus.LANG_NAMES[v]} speech", (lambda v: lambda c: c["version"] == v)(v))
                            for v in VERSIONS]:
            out += [f"**{title}**", "", "| Whisper \\ LLM | " + " | ".join(models) + " |",
                    "|---|" + "---|" * len(models)]
            for s in sources:
                cells = []
                for m in models:
                    d = llm.get((qp, m, s))
                    st = quality_stats([c for c in _cases(d) if pred(c)]) if d else None
                    cells.append("–" if not st or not st["n"] else
                                 f"{_pct(st['facts'])}" + (f" ⚠{st['invented']}" if st["invented"] else ""))
                out.append(f"| {s.split('/')[1]} ({s.split('/')[0]}) | " + " | ".join(cells) + " |")
            out.append("")
        out.append("⚠n = n summaries with invented / forbidden content.")

    # --- laptop speed
    lap = cpu_speed(llm)
    if lap:
        out += ["", "## 4. Summary speed without a GPU", "",
                "All summaries the model wrote on this profile (every transcript source).", "",
                "| LLM | Profile | Summaries | Time per summary | Slowest | First text after | Tokens/s | RAM |",
                "|---|---|---|---|---|---|---|---|"]
        for (p, m), st in sorted(lap.items(), key=lambda kv: (kv[0][0], kv[1]["seconds"] or 1e9)):
            ok = st["ok"]
            out.append(f"| {m} | {p} | {len(ok)} | {_mean([c['seconds'] for c in ok]) or 0:.0f} s | "
                       f"{max([c['seconds'] for c in ok], default=0):.0f} s | "
                       f"{_mean([c['first_token_s'] for c in ok]) or 0:.0f} s | "
                       f"{_mean([c['tokens_per_s'] for c in ok]) or 0:.1f} | {st['size_mb'] / 1024:.1f} GB |")

    # --- recommendation
    out += ["", "## 5. Best measured combination per machine", "",
            f"Highest share of facts among the combinations that fit (penalised for invented content / wrong "
            f"language). GPU budgets leave room for the CUDA context; CPU-only needs Whisper RTF ≤ {MAX_CPU_RTF} "
            f"and a summary within {MAX_CPU_SUMMARY_S} s.", "",
            "| Machine | Whisper | LLM | Why |", "|---|---|---|---|"]
    out += [f"| {t} | {w} | {m} | {why} |" for t, w, m, why in recommend(stt, llm)]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("stt", help="transcribe every recorded corpus script with each Whisper model")
    a.add_argument("--models", nargs="+", required=True)
    a.add_argument("--profiles", nargs="+", default=["gpu", "laptop"], choices=list(PROFILES))
    a.add_argument("--force", action="store_true")
    b = sub.add_parser("llm", help="summarise transcripts with each LLM and score them")
    b.add_argument("--models", nargs="+", required=True)
    b.add_argument("--sources", nargs="+", default=["reference"],
                   help="reference (the script), gpu/<whisper>, laptop/<whisper>, or all")
    b.add_argument("--profile", default="gpu", choices=list(PROFILES))
    b.add_argument("--scripts", nargs="*", help="limit to these corpus ids")
    b.add_argument("--langs", nargs="+", default=["en", "ar"])
    b.add_argument("--force", action="store_true")
    c = sub.add_parser("mem", help="measure each LLM's real GPU / RAM use while it writes a summary")
    c.add_argument("--models", nargs="+", required=True)
    c.add_argument("--profiles", nargs="+", default=["gpu", "laptop"], choices=list(PROFILES))
    c.add_argument("--force", action="store_true")
    r = sub.add_parser("report", help=f"write {REPORT.relative_to(ROOT)} (or --out)")
    r.add_argument("--out", type=Path, default=REPORT,
                   help="report file, e.g. docs/BENCHMARK_LAPTOP.md on the demo laptop (keeps the dev-PC report)")
    w = sub.add_parser("_stt")                    # internal: one Whisper model in a fresh process
    w.add_argument("model")
    w.add_argument("profile")
    args = ap.parse_args(argv)
    if args.cmd == "stt":
        return run_stt(args)
    if args.cmd == "llm":
        return run_llm(args)
    if args.cmd == "_stt":
        return stt_worker(args.model, args.profile)
    if args.cmd == "mem":
        return run_mem(args)
    args.out.write_text(render_report(), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
