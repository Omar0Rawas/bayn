# Progress log

## Plan (approved 2026-11-08)
1. Repo, simulated EEG, UI skeleton
2. Connect the real Muse 2 (BrainFlow `MUSE_2_BOARD`, muse-lsl fallback) — becomes the default source
3. Gesture trigger + events (`Detector` interface, clench / double blink, calibration, cooldown, EEG window −10 s/+30 s, manual button + hotkey)
4. Whisper live transcription (faster-whisper, live mic, en / ar / mixed)
5. Ollama summaries, transcript → summary view in the UI
   (+ model benchmark 2026-11-10: default summary model gemma4:e2b / qwen3:4b, re-test on the demo laptop)
6. Triple-blink trigger + ML detector, EEG only (revised 2026-11-10: three blinks instead of two, no accelerometer,
   onboarding dropped)
7. Setup & packaging polish (Docker dropped 2026-11-09: Bluetooth, mic and GPU all need native processes):
   one-command setup incl. Ollama check + model pulls, offline-readiness check, laptop checklist, docs
8. Full end-to-end test run with simulated and real data

## Phase 1 — repo, simulated EEG, UI skeleton ✅
Built:
- Config system (YAML + env overrides), ZeroMQ bus, typed messages.
- `SimulatedEEG`: pink-noise background, alpha/theta, mains hum, DC drift, random natural blinks and weak EMG bursts as
  distractors; injectable `clench`, `double_blink`, `blink`. Deterministic with a seed.
- Streaming 1–40 Hz bandpass + notch (`auto` = 50 and 60 Hz), ring buffer, MAD-based channel quality, mains detection.
- EEG bridge process; FastAPI dashboard with live canvas plot, quality chips, simulator buttons, placeholders for later phases.
- `python -m bayn.run --demo` launcher, `bayn.doctor` machine check, `scripts/bayn.ps1`, Makefile, `scripts/claude-session.ps1`.
- 23 tests, all passing.

Decisions / gotchas:
- pyzmq asyncio sockets fail on Windows' default event loop → dashboard reads ZeroMQ in a thread (`ThreadedSubscriber`).
- uv not installed on the dev PC → plain venv + pinned requirements.
- Python ≥ 3.12 required.

## Phase 2 — real Muse 2 ✅
Built:
- `BrainFlowSource` (default `eeg.source: muse`): connects in a worker thread, auto-reconnects when data stops for 3 s,
  reports progress ("searching for Muse-3EE1...") to the dashboard. Any BrainFlow board via `eeg.brainflow.board`.
- `LSLSource` (fallback): reads `muselsl stream`, picks channels by label, converts LSL clock to unix time.
  `run --source lsl` starts and restarts `muselsl stream` automatically.
- `SampleClock`: regular timeline from jittery device timestamps, re-anchors after gaps; bridge resets filters after gaps.
- `bayn check` (10 s connection + quality report), Bluetooth/Muse scan in `doctor`, pairing + troubleshooting in README.
- `bayn.cmd` wrapper: Windows blocks .ps1 scripts by default; the wrapper bypasses that per call.
- Brainwave panel (user request): delta/theta/alpha/beta/gamma share of power per sensor, labelled like "Alpha (AF7)",
  computed in the bridge at 2 Hz (`eeg.bands` topic, also reusable as EEG stats for Phase 5 summaries).
- Tests reorganised by level with pytest markers: 32 unit, 6 integration, 2 system (real processes end to end),
  2 hardware (real Muse, opt-in via BAYN_HARDWARE=1). Full plan, results and defect log: docs/TESTING.md.

Hands-on results (dev PC, Muse-3EE1, 2026-11-08):
- BrainFlow: connects in ~6-12 s, 256.0 Hz, no drops, all channels "good" (13-30 µV). Mains detected: **60 Hz**.
- muse-lsl: same quality, 256.0 Hz.
- Full launcher with the real headset: "searching" -> connected -> live plot within ~12 s.

Decisions / gotchas:
- Dev PC built-in Bluetooth (QCA61x4A, no antennas) could not see the Muse → USB dongle (Actions ATS2851). On Windows 10
  it showed Code 31 until its vendor driver was installed; the built-in adapter is disabled (Code 22). Laptop: run `doctor`.
- muselsl 2.5.3 crashes with bitstring 5 → pinned bitstring 4.4.0.
- muse-lsl stamps samples with unix time (not the LSL clock) → clock style detected per stream.
- Hardware tests: BrainFlow 256.0 Hz, latency median 20 ms; muse-lsl 256.0 Hz, latency median −23 ms.
- muse-lsl pulls pandas/scikit-learn/matplotlib → kept in requirements/muselsl.txt (top-level pins, cross-platform).

## Phase 3 — gesture trigger + events ✅
User test with Muse-3EE1 (2026-11-09): jaw clench works before and after calibration. Rule-based double blink
false-triggers on head movement and misses natural blinks (TESTING.md F1) → ML detector in Phase 6;
**jaw clench is the recommended trigger until then.**

Built:
- `Detector` interface (`events/detectors/base.py`) with `ClenchDetector`, `DoubleBlinkDetector`, `ManualDetector`,
  `SeizureDetectorStub` (documented plug-in point for a real model). Gesture chosen by `events.gesture`
  (clench | double_blink | both | manual), switchable at runtime from the dashboard.
- Clench: 55-100 Hz RMS envelope on TP9/TP10 (mains + harmonics notched), must hold 1 s, adaptive threshold
  (4x resting level) until calibrated. Double blink: signed peaks on the AF7/AF8 average (0.3-10 Hz), exactly two,
  0.2-0.8 s apart.
- `EventManager` (pure, data-time driven): cooldown 15 s, manual debounce 2 s, −10/+30 s window saved to
  `data/events/<id>/` (eeg.npz, eeg.csv, meta.json with before/after band stats) + SQLite index `data/bayn.db`.
- Guided calibration: 15 s rest + 3 prompted reps; threshold = geometric mean of rest and gesture level; saved per
  user AND per source in `data/profiles/<profile>-<source>.json` (so sim calibration never overwrites the real one).
  In sim mode the simulator performs the prompted gestures.
- `events` service process (ports 5560 pub / 5561 cmd), started by `bayn run` / `bayn demo`.
- Dashboard: trigger status (armed / cooldown / calibrating), gesture selector, live level-vs-threshold meters,
  working Trigger button + `T` hotkey, sound toggle (Web Audio beep), event flash banner + plot marker,
  calibration overlay with prompts and results, event timeline with CSV/NPZ downloads.
- Logo: README header + dashboard emblem/favicon; dashboard accent colours from the logo.

Results (simulated, with distractors): clench 40/40, 0 false in 20.7 min; double blink 58/60 (97 %), 0 false in
62 min; onset error ≤ 0.14 s. 74 automated tests pass (63 unit, 6 integration, 5 system) + 2 hardware.

Decisions / gotchas:
- Double blink needed signed peaks: the band-pass rebound after every blink looked like a 3rd blink (D9).
- Events right after start have a short pre-window → flagged `partial` (D10). Demo listens to both gestures (D11).
- Thresholds on the real headset are untested defaults until the user calibrates; Phase 6 runs the formal protocol.

## Phase 4 — Whisper live transcription ✅
Built:
- `audio` process: live microphone (sounddevice; default input or `audio.device`) or WAV file played in real time;
  16 kHz mono on the bus + 10 Hz mic level + status; reopens the mic if it disappears.
- `transcriber` process: Silero VAD segmenter (0.6 s pause ends an utterance, 15 s max) → faster-whisper.
  auto: large-v3-turbo on CUDA (float16, beam 5, live partials) / small on CPU (int8, beam 1); warm-up + CPU fallback;
  degraded mode if no model (EEG side keeps working). Language `mixed` (ar/en per utterance) | en | ar | auto,
  switchable live. Hallucination filter. Segments in SQLite (`segments` table), unix timestamps.
- Event hand-off: on `events.done` the transcript of the window is attached (`transcript.txt/json` in the event folder,
  event row updated, `transcript.window` published) → input for Phase 5.
- Dashboard: transcript card (model/device badge, language selector, mic meter + device, segments with time +
  language tag, RTL Arabic, live partial line, segments inside event windows highlighted), transcript link per event.
- `bayn models` (download once, offline after), `bayn mics`, `bayn sample-audio` (Windows TTS: English + Arabic
  "Naayf" → samples/audio/demo.wav + tests/data clips), `--audio file`, `doctor` checks model/CUDA/mic.
- requirements/stt.txt (pinned top-level), requirements/gpu.txt (CUDA 12 + cuDNN 9 wheels, only with NVIDIA).

Results: GPU large-v3-turbo 0.46 s/utterance, 14/14 ar↔en switches; CPU small 2.4 s/utterance; end to end on GPU text
appears 1.34 s after speech ends. 93 tests (76 unit, 9 integration, 6 system, 2 hardware).

Decisions / gotchas: Docker dropped. `self.step` shadowed `step()` (D12). PowerShell 5.1 reads .ps1 as ANSI → Arabic in
a UTF-8 JSON (D13). Audio is never saved as sound, only text.

User live-mic test (2026-11-09, classroom scripts): English 1.2 % WER, Arabic (Hejazi) 28.8 % WER / 12 % CER,
intra-sentence mixed 39.7 % WER (English-decoded sentences translate the Arabic) → accepted as a **current limitation**
(TESTING.md F2). Scripts + live transcripts saved as tests/data/scenarios/classroom.json (also Phase 5 summary tests).
Added: sentence tail for event transcripts (D15), WER/CER metrics, `bayn record`, `bayn eval-stt`, optional
`mixed_strategy: arabic_lean` and `initial_prompt: auto` (off by default until benchmarked).

**Later (user request):** user writes more realistic classroom scripts (mostly Arabic with English key terms such as
"pie", "Rutherford", which must not be lost); record them once and benchmark ALL Whisper models × devices × strategies
on the same audio files (WER, CER, key-term recall, latency) to understand limits and pick defaults per machine.

## Phase 5 — local LLM summaries ✅ (user-tested 2026-11-09: live capture + summaries "very satisfactory")
Built:
- Ollama installed on the dev PC (winget); `bayn setup` installs it (Windows) and pulls the model; `bayn models`.
- `summarizer` process (5580 pub / 5581 cmd): on `transcript.window` → structured summary via Ollama
  (`format` = JSON schema: summary, ≤ 4 key points, unclear) → rendered by code with localised headings → streamed
  to the dashboard as it is written (partial JSON rendering) → summary.md/json in the event folder. Model preloaded at
  start; one retry; 60 s stall timeout; no-speech windows skip the LLM; regenerate + language (auto/en/ar/both).
- Prompts in the output language (Arabic instructions for Arabic output), rules against inventing / repeating
  recognition errors, worked example on an unrelated topic. EEG section written by code (not the LLM).
- `model: auto` → qwen2.5:7b with NVIDIA ≥ 10 GB (or Apple Silicon ≥ 16 GB), else qwen2.5:3b.
- Dashboard Summaries card: ① heard → ② context (EEG + exact prompt) → ③ summary writing live; EN/AR tabs.
- `bayn eval-summary`: automatic quality checks on the classroom scenarios (facts, invented incl. meaning flips,
  language, format), model comparison, `--rescore`.

Results: qwen2.5:7b structured: facts 96 % (en) / 68 % (ar), 0/12 invented, 3.3 s. llama3.2:3b rejected (invented
content, wrong language, loops). End to end on GPU: summary 2.4 s after the window closed. 121 tests (97 unit,
16 integration, 6 system, 2 hardware).

Decisions / gotchas: structured output beat free markdown (fixed Arabic format, stopped line-by-line translation,
avoided a "level rose" meaning flip). Parked LLMs + Whisper oversubscribed the GPU (D17) → stall timeout, evaluation
unloads models, 7b only with ≥ 10 GB.

Follow-up (user requests, 2026-11-09):
- **Laptop is the end goal** (CLAUDE.md, memory): every change checked for CPU-only, VRAM budget, disk, offline.
- **Disk**: dev PC C: had 5.2 GB free. Ollama models moved to models/ollama (OLLAMA_MODELS), pip cache purged,
  llama3.2:3b removed → 14.7 GB free. Setup defaults to this; `doctor` shows sizes + free space; `bayn run` starts
  `ollama serve` itself (tray app unreliable) with OLLAMA_NO_CLOUD=1 (Ollama 0.40 had cloud features on).
- **Recording corpus**: docs/CORPUS.md (tests/data/corpus.json): 12 scripts, 4 topics × en / ar / mixed;
  `bayn record <id>`, `bayn eval-stt --corpus all [--models ...] [--device cpu]` with key-term recall.
  Display only: one sentence per line, English words inside Arabic isolated + bold (right-to-left layout moved
  them); `bayn record` / `bayn corpus show <id>` open an offline reading page (terminals can't render Arabic).
  All 12 scripts recorded by the user (data/recordings, not committed).

User test: live mic capture + summaries in the dashboard confirmed working well. Phase 5 committed.

## Model benchmark (unplanned phase between 5 and 6, user request 2026-11-10) ✅ dev PC · ⏳ demo laptop
Goal: the strongest Whisper × LLM combination for English / Arabic / mixed, and proof it still works with minimal
resources. Details: docs/TESTING.md ("Model benchmark"), full tables: docs/BENCHMARK.md.
Built:
- `bayn benchmark stt | llm | mem | report` (src/bayn/benchmark.py): every Whisper model on the user's 12 corpus
  recordings, every LLM on perfect and real transcripts, CPU-only speed, measured GPU/RAM, per-machine
  recommendation. Profiles: `gpu`, `laptop` (CPU, 4 threads: emulated minimal laptop), `cpu` (all cores: for the
  real laptop). Cached + resumable in data/eval/bench (gitignored); scores recomputed from saved text.
- Fair scoring: numbers in words = digits (en + ar), alternative spellings for key terms, transliterated English
  terms accepted in Arabic summaries, language check ignores the quoted "Unclear" fragments (D19–D22).
- `transcription.cpu_threads` setting; Ollama client: per-call options (CPU only), `think: false` for reasoning
  models, `local_server()` used by every tool (never the `ollama` CLI: it launches the tray app, E6).
Results (dev PC): Whisper defaults confirmed (turbo on GPU; small is the only accurate model that keeps up live on
4 CPU threads, RTF 0.36). Summary model changed: **gemma4:e2b** (81 % facts on real turbo transcripts, 82 % on mixed
speech, 3.0 GB VRAM, 8 s per summary on 4 CPU threads / 4.7 GB RAM) when NVIDIA ≥ 6 GB or RAM ≥ 12 GB, else
**qwen3:4b** (3.1 GB RAM). Old defaults qwen2.5:7b / 3b scored 68 % / 44–52 %. Mixed speech: Whisper writes English
key terms in Arabic letters (F4); Gemma 4 still understands them.
Disk: models/ now holds every candidate (Ollama 33 GB, Whisper 6.5 GB) on H:. Copy only what the laptop needs
(`bayn models` there pulls just the auto choice); unused candidates can be deleted after the laptop retest.

**Re-test on the demo laptop** (user, 2026-11-10): the best demo configuration is decided there, with the laptop's
own (mini) microphone: record the corpus again, then
`bayn benchmark stt --models tiny base small medium large-v3-turbo --profiles cpu` (+ `gpu` if NVIDIA),
`bayn benchmark llm --profile cpu --sources all --models gemma4:e2b qwen3:4b`, `bayn benchmark mem --profiles cpu ...`,
`bayn benchmark report`. The user will ask for a prompt for the Claude Code session on the laptop.

Next (user's order, 2026-11-10): Phase 6 blink detection (ML double blink) → if all good, move to the laptop
(setup + retests) → clean up the repo and make it presentable.

## Phase 6 — triple-blink trigger + ML detector (EEG only) ✅ (user-tested 2026-11-10: satisfactory for a demo)
Scope (user, 2026-11-10): EEG only (the end device won't have an accelerometer), onboarding dropped, use the
synchronised TP9/TP10 peak the user saw when blinking.
Built:
- Labelled recording sessions (`events/training.py`): standard (~4.7 min: double/single blinks, head turns, nods,
  sideways glances, talking, random order) and noisy (~4.5 min: double blinks during talking / looking around /
  moving / sitting). Dashboard: Event trigger → Blink detector → Train… Saved to data/training/<profile>-<source>/.
- `events/blink_ml.py`: sensitive streaming candidate finder (forehead peaks → episodes) → 27 EEG features
  (forehead shape, AF7/AF8 agreement, TP9/TP10 vs forehead, slow drift, muscle energy) → random forest per user →
  `MLDoubleBlinkDetector` (rule fallback; `events.double_blink.method: auto|rule|ml`). Held-out evaluation vs the
  rule: leave one session out / time blocks. CLI: `python -m bayn.events.blink_ml train|inspect`.
- Training runs in a background thread in the events service; the new model is used immediately.
- Simulator: nod, head turn, sideways glance, talking artefacts + TP blink coupling; dashboard buttons for them.
- scikit-learn added to the pinned base requirements (CPU only, model ~1 MB: fine for the laptop).
**Trigger changed to THREE blinks** (user, 2026-11-10, before any real data was recorded): double and single blinks
are now near misses that must not fire (recorded as negatives in the standard session). The rule detector counts
N blinks (`events.triple_blink.count`) and was fixed for fast series (D24: later blinks ride on the high-pass
rebound of the previous one -> counted by their rise from the dip). Simulated nodding now nods 2-3 times.
Results (simulator, 4 sessions / 19 min, leave one session out): rule 96 % caught, 1.08 false triggers/min
(mostly nodding three times); ML 95 % caught, 0.34/min. With only 2 sessions the ML trailed the rule (82 %,
0.95/min): it needs ~2 standard + 2 noisy sessions. Auto only uses ML once it beats the rule held-out. Real data so far (Phase 3 windows, unlabelled): blinks 300–450 µV / ~170 ms wide; TP9/TP10 weakly
positive with the forehead peak; forehead noise 22–87 µV → finder threshold capped at 100 µV (D23).
Tests: 196 (165 unit, 22 integration, 7 system, 2 hardware).

**First real sessions (2026-11-10, standard + noisy):** "still not very accurate": rule 12/28, ML 10/28. Plots showed
why: on the Muse, blinks are largest on TP9/TP10 (forehead reference), AF8 is noisy, nods swing AF7/AF8. Fixes:
a per-person learned blink signal (rule + finder + features), and synthetic training data recombined from the
person's own blinks (user's idea; training only). Held-out on real data: ML 23/28 (82 %), 0.81 false/min (was
36 %, 2.3/min); learned-signal rule 79 %, 2.16/min. New `boundaries` protocol (user's idea): soft/fast/slow triple
blinks and the closest look-alikes (four blinks, squeeze, nod three times, look up/down, eyebrows).

**Decision test: is ML necessary?** (user's question once the signal was right; pre-registered 2026-11-10;
docs/BLINK_DECISION.md via `.\bayn blink decide`). Rule on learned signal vs classifier (real + synthetic),
leave one session out, real data only. ML counts as necessary only if clearly better (>= +10 points caught with
no more false triggers, or >= 0.5/min fewer false triggers at <= -5 points). Final once there are standard +
noisy + boundaries sessions, seated and standing (the user offered standing / moving / re-recording; walking in
place gets spoken prompts; sessions can be deleted and re-recorded from the dashboard). Preliminary (2 seated
sessions): rule 86 % / 0.41 false/min vs ML 82 % / 0.81 -> ML NOT necessary so far. Fixed D26 on the way
(threshold fell back to 90 uV in folds trained on the noisy session).

**FINAL decision** (6 sessions, 31 min, seated + standing): rule on the learned signal 67 % caught / 0.30 false per
min vs ML 64 % / 0.73 -> **ML is not necessary**; Auto runs the rule. Recall is limited by the rule's fixed
0.8 s gap limit (slow/soft triple blinks) and by the first blink of a series measuring smaller; optional Rule v2
(learned timing) noted in TESTING.md. User demo test: intentional, not-too-soft triple blinks are caught almost
every time -> satisfactory for the demo. Dashboard option renamed: "Basic rule (untrained forehead average)".

Demo tips: three firm blinks at a steady ~2/s, head still; event ~0.5 s after the third blink; 15 s cooldown.

**Superseded plan (kept for history): the user's headset sessions** (they offered more sessions, noisy ones too): calibrate the triple blink,
then Train… → 2 standard + 2 noisy sessions (~20 min); re-seat the headset between sessions; more sessions on other
days improve robustness. Then compare rule vs ML on the held-out report, 10 min normal-activity false-trigger test.
After that: move to the laptop (setup + benchmark retest), then repo cleanup.

