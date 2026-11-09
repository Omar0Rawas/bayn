# Bayn — notes for Claude sessions

Local hackathon demo: Muse 2 EEG -> on-command gesture trigger -> Whisper live transcript -> Ollama summary.
Prototype, NOT a medical device. Current status and next steps: docs/PROGRESS.md (update it at the end of every phase).

## END GOAL: it must run fine on the demo LAPTOP (user, 2026-11-09)
Check every change against the laptop, not just the dev PC:
- CPU-only path works (Whisper small, gemma4:e2b / qwen3:4b) and stays real-time; nothing assumes CUDA.
- GPU memory budget: Whisper turbo (~2.3 GB) + LLM must fit together (gemma4:e2b measured 3.0 GB VRAM).
- The model benchmark (docs/BENCHMARK.md) ran on the dev PC; the **demo configuration is re-tested on the laptop**
  with its own mic (`bayn benchmark ... --profiles cpu`, see docs/PROGRESS.md) before the hackathon.
- **Disk space**: laptops often have a small C: drive. All models live in the repo's models/ folder
  (models/whisper, models/ollama via OLLAMA_MODELS) and can be copied to the laptop; `doctor` shows sizes + free space.
- Setup from scratch with one command; offline after setup; `doctor` must flag anything missing.
- Don't rely on background apps (e.g. the Ollama tray app): `bayn run` starts what it needs.

## Working agreement
- Work in the agreed phases (see docs/PROGRESS.md). **Stop after each phase** so the user can test; commit only after they confirm.
- Developed on a desktop PC (Windows 10, RTX 2080 Ti), demoed on a different laptop with unknown hardware:
  **never hardcode hardware**. GPU/CPU, Whisper model, mains frequency are auto-detected.
- After Phase 2 the **real Muse 2 is the default EEG source**; simulated stays available via `--demo` / `--source sim`.
  Don't run anything that needs the headset unless the user says it's connected.
- 100% offline after setup: no cloud APIs, no telemetry, no CDN assets in the UI.
- Demo mode uses the live microphone (WAV file source is optional).
- Languages: English, Arabic, and mixed (per-utterance language detection limited to {ar, en}).

## Architecture
- Separate processes talk over ZeroMQ (TCP) — msgpack frames `[topic, payload]`, numpy arrays as raw bytes (src/bayn/common/bus.py).
  Topics/messages: src/bayn/common/schemas.py.
- Do NOT use `zmq.asyncio` in the FastAPI server: it fails on Windows' Proactor loop. Use `ThreadedSubscriber`.
- EEG bridge publishes both `eeg.raw` (needed for clench/EMG detection at 55-100 Hz) and `eeg.filt` (1-40 Hz + notch).
- Processes: eeg bridge (5555 pub / 5556 cmd), events service (5560 pub / 5561 cmd), audio (5570 pub),
  transcriber (5575 pub / 5576 cmd), dashboard (8080). `bayn.run` starts all of them; audio + transcriber are
  restarted if they die and the transcriber has a degraded mode (no model → error status, rest keeps running).
- Summarizer (5580 pub / 5581 cmd): Ollama at 127.0.0.1:11434 (installed on dev PC at
  %LOCALAPPDATA%\Programs\Ollama). Structured output (JSON schema) rendered by code; EEG section written by code,
  never by the LLM. model auto = gemma4:e2b if NVIDIA >= 6 GB / Apple Silicon / RAM >= 12 GB, else qwen3:4b
  (chosen by `bayn benchmark`; old qwen2.5 7b/3b scored lower). `think: false` is sent (reasoning models).
  Unload models you load for experiments (keep_alive 0) — parked models + Whisper stalled the GPU once (D17).
  Measure prompt/model changes with `bayn benchmark` (corpus recordings) or `bayn eval-summary`, never by eye.
  Never call the `ollama` CLI (it launches the tray app, E6): use `local_server()` from ollama_client.
  Ollama's /api/ps under-reports Gemma 4 memory: measure from outside (`bayn benchmark mem`).
- **No Docker** (dropped by the user): Bluetooth, mic and GPU all need native processes.
- Whisper: models/whisper/<name>/ loaded with local_files_only; downloaded only by `bayn models`. auto = large-v3-turbo
  on CUDA (float16, beam 5, live partials), small on CPU (int8, beam 1). CUDA DLLs from pip nvidia-* wheels are added
  via os.add_dll_directory on Windows. Warm-up inference catches cuDNN problems → CPU fallback.
- Transcription runs continuously; on `events.done` it attaches transcript.txt/json to the event (→ Phase 5 input).
- Windows PowerShell 5.1 reads .ps1 files as ANSI: keep scripts ASCII; put Arabic text in UTF-8 data files. Events logic lives in `EventManager` (pure, data-time driven → test offline
  with tests/helpers.py `sim_chunks`). Event stream: `events.event` → `events.done` (window saved: starts pipeline).
- Data: data/events/<id>/ (eeg.npz/csv, meta.json), data/bayn.db (SQLite), data/profiles/<profile>-<source>.json.
- Config: config/default.yaml <- config/local.yaml <- .env <- env vars `BAYN_SECTION__KEY` (src/bayn/common/config.py).

## Hardware notes
- User's headset: Muse-3EE1. Mains on the dev PC's grid is 60 Hz (auto-detected; do not hardcode).
- Dev PC uses a USB Bluetooth dongle; its built-in adapter is disabled. Only one BLE adapter may be active.
- muselsl needs bitstring<5.

## Commands (Windows)
- `.\bayn setup | demo | run | check | models | mics | corpus | record | eval-stt | eval-summary | benchmark | sample-audio | test | doctor`
  (`--audio file` = no mic)
- Recording corpus: tests/data/corpus.json (source) → docs/CORPUS.md (`python -m bayn.corpus render`; a test fails if
  stale). 4 topics × en / ar / mixed (Arabic + English key terms). `bayn record <id>` (opens an RTL reading page;
  terminals can't render Arabic). `bayn benchmark stt|llm|mem|report` = Whisper × LLM benchmark, profiles gpu /
  laptop (4 CPU threads) / cpu (all cores), cached in data/eval/bench, report → docs/BENCHMARK.md. Metrics treat
  numbers in words = digits; key terms accept `a|b` spellings. Recordings in data/recordings (never committed). (bayn.cmd wraps scripts/bayn.ps1 with ExecutionPolicy Bypass,
  because Windows blocks .ps1 scripts by default)
- Tests: `.\.venv\Scripts\python.exe -m pytest` (no hardware needed); levels via `-m unit|integration|system`;
  real headset: `$env:BAYN_HARDWARE=1; ... -m hardware -v -s`.
- Keep docs/TESTING.md current every phase: per-phase results, traceability matrix, defect log (D#), env issues (E#).
- Python 3.12+ (numpy 2.5 dropped 3.11). Deps pinned in requirements/*.txt; pyproject has loose ranges.
