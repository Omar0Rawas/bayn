# Progress log

## Plan (approved 2026-11-08)
1. Repo, simulated EEG, UI skeleton
2. Connect the real Muse 2 (BrainFlow `MUSE_2_BOARD`, muse-lsl fallback) — becomes the default source
3. Gesture trigger + events (`Detector` interface, clench / double blink, calibration, cooldown, EEG window −10 s/+30 s, manual button + hotkey)
4. Whisper live transcription (faster-whisper, live mic, en / ar / mixed)
5. Ollama summaries (qwen2.5:3b default), transcript → summary view in the UI
6. Tune gesture triggering on the real headset + guided onboarding (user request, "fun" feature): Muse-app-style
   walkthrough ("blink twice", "clench your jaw", "close your eyes and watch alpha rise") that doubles as calibration
   + **ML double-blink detector** (user request after the Phase 3 headset test): the guided onboarding records
   labelled data → features → random forest (double / single blink / movement / other) → `MLDoubleBlinkDetector`;
   two variants compared: EEG only (like a future device) and EEG + Muse accelerometer; rule-based stays as fallback.
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

Next: Phase 5 (Ollama summaries).
