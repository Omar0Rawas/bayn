# Bayn — notes for Claude sessions

Local hackathon demo: Muse 2 EEG -> on-command gesture trigger -> Whisper live transcript -> Ollama summary.
Prototype, NOT a medical device. Current status and next steps: docs/PROGRESS.md (update it at the end of every phase).

## END GOAL: it must run fine on the demo LAPTOP (user, 2026-11-09)
Check every change against the laptop, not just the dev PC:
- CPU-only path works and keeps up; nothing assumes CUDA. **Demo laptop tested 2026-11-10** (ASUS Vivobook 15,
  i7-1355U 15 W, 8 GB single-channel RAM, no NVIDIA GPU; user: "satisfactory"): Whisper small + single-encode,
  gemma4:e2b, triple blink; text 6-10 s after speech, summary ~30 s after the window. Results:
  docs/BENCHMARK_LAPTOP.md and docs/PROGRESS.md "Demo laptop session" (incl. what to do next on the dev PC).
- **Memory is the laptop's bottleneck**, not CPU: with Bayn + gemma4:e2b loaded an 8 GB machine has ~1-2 GB left;
  swapping made Whisper 2-6x slower (D30). Don't add per-process memory (OPENBLAS_NUM_THREADS=1 is set by `bayn run`);
  the demo runs with only Bayn + one light browser tab open. The 15 W chip is power-limited: more threads don't help.
- GPU memory budget: Whisper turbo (~2.3 GB) + LLM must fit together (gemma4:e2b measured 3.0 GB VRAM).
- Model benchmarks: dev PC → docs/BENCHMARK.md, demo laptop (`--profiles cpu`) → docs/BENCHMARK_LAPTOP.md
  (`bayn benchmark report --out ...`; never overwrite the other machine's report).
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
  transcriber (5575 pub / 5576 cmd), showcase driver (5590 pub, only in `bayn showcase`), dashboard (8080). `bayn.run` starts all of them; audio + transcriber are
  restarted if they die and the transcriber has a degraded mode (no model → error status, rest keeps running).
- Summarizer (5580 pub / 5581 cmd): Ollama at 127.0.0.1:11434 (installed on dev PC at
  %LOCALAPPDATA%\Programs\Ollama). Structured output (JSON schema) rendered by code; EEG section written by code,
  never by the LLM. model auto = gemma4:e2b if NVIDIA >= 6 GB / Apple Silicon / no NVIDIA GPU and RAM >= 7 GB /
  small NVIDIA GPU and RAM >= 12 GB, else qwen3:4b (chosen by `bayn benchmark` on both machines; on the laptop
  gemma4:e2b was 1.7 GB resident, 19 s/summary, 70 % facts vs qwen3:4b 35 s, 45 %). `think: false` is sent.
  Unload models you load for experiments (keep_alive 0) — parked models + Whisper stalled the GPU once (D17).
  Measure prompt/model changes with `bayn benchmark` (corpus recordings) or `bayn eval-summary`, never by eye.
  Never call the `ollama` CLI (it launches the tray app, E6): use `local_server()` from ollama_client.
  Ollama's /api/ps under-reports Gemma 4 memory: measure from outside (`bayn benchmark mem`).
- **No Docker** (dropped by the user): Bluetooth, mic and GPU all need native processes.
- Whisper: models/whisper/<name>/ loaded with local_files_only; downloaded only by `bayn models`. auto = large-v3-turbo
  on CUDA (float16, beam 5, live partials), small on CPU (int8, beam 1). CUDA DLLs from pip nvidia-* wheels are added
  via os.add_dll_directory on Windows. Warm-up inference catches cuDNN problems → CPU fallback.
  Every utterance costs a full 30 s Whisper window, however short. `mixed` normally encodes twice (detect_language +
  transcribe); `SingleEncodeWhisperEngine` (engine_single.py) reuses the detection pass → identical text, 26-48 %
  less time on CPU. `transcription.single_encode: auto` = on for CPU, off for CUDA until
  `python -m bayn.transcription.regress_single --device cuda` passes on the dev PC (pending). Engine comparisons
  must decode at temperature 0 (`regress_single.greedy`): faster-whisper's fallback samples randomly (E9).
- Transcription runs continuously; on `events.done` it attaches transcript.txt/json to the event (→ Phase 5 input).
- Windows PowerShell 5.1 reads .ps1 files as ANSI: keep scripts ASCII; put Arabic text in UTF-8 data files. Events logic lives in `EventManager` (pure, data-time driven → test offline
  with tests/helpers.py `sim_chunks`). Event stream: `events.event` → `events.done` (window saved: starts pipeline).
- Data: data/events/<id>/ (eeg.npz/csv, meta.json), data/bayn.db (SQLite), data/profiles/<profile>-<source>.json.
- Trigger gesture = TRIPLE blink (user, 2026-11-10; `events.triple_blink.count: 3`; double/single blinks must not
  fire). Phase 6: EEG ONLY (no accelerometer: the end device won't have one). `events/training.py` =
  labelled recording sessions (standard / noisy) → data/training/<profile>-<source>/*.npz; `events/blink_ml.py` =
  candidate finder → 27 features (incl. TP9/TP10 vs forehead) → random forest → data/models/blink-<profile>-<source>
  .joblib → `MLBlinkDetector` (rule fallback; `triple_blink.method` auto = ML only if it beat the rule held-out; it did not: docs/BLINK_DECISION.md). Finder + features must stay identical
  offline and live (test). Real Muse forehead noise is 22-87 µV (sim ~10): check changes on real windows too.
- Blink pattern (2026-11-11), most personal first: trained model > Calibrate-learned (profile `weights`) > built-in
  Muse pattern (events/assets/blink_muse.json, real headsets only) > AF7/AF8 "front" (simulator). `.\bayn blink
  calibration` = held-out check on real sessions (Calibrate 66 %/0.52 vs trained 67 %/0.30 per min).
- Seizure socket (2026-11-10): `SeizureModelDetector` (events/detectors/seizure_model.py) = MOCK logistic model
  trained on CHB-MIT (bayn.seizure: chbmit / features / train; assets/model.json + chbmit_clip.npz shipped; data in
  data/datasets, gitignored). Always say: NOT validated on Muse data, NOT for clinical use (docs/SEIZURE_MODEL.md).
  Off by default (`events.seizure.enabled`), on in `bayn demo`; simulator "seizure" replays the clip.
- Showcase (2026-11-11): `bayn showcase` = sim EEG + samples/showcase/*.wav (user's wireless-mic corpus, public OK) +
  driver bayn.showcase (5590 pub, topic showcase.scene → dashboard banner). Preview: `python -m bayn.preview record |
  export | gif` → docs/preview (static replay, works from file://) + docs/media/*.gif (headless Edge, no new deps).
  Re-record after UI changes. Seizure alarms bypass the gesture cooldown (D32); the seizure detector survives
  gesture changes (D33).
- README is for judges first (About → Try it → Built with …); keep the warnings and the "mock / not validated /
  not an absence-seizure detector" wording wherever the seizure model appears. Edit docs with the Edit tool or byte-safe
  scripts: `.\bayn` in a non-raw Python string becomes a backspace (D34).
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
