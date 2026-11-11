# Progress log

## Plan (approved 2026-11-08)
1. Repo, simulated EEG, UI skeleton
2. Connect the real Muse 2 (BrainFlow `MUSE_2_BOARD`, muse-lsl fallback) — becomes the default source
3. Gesture trigger + events (`Detector` interface, clench / double blink, calibration, cooldown, EEG window −10 s/+30 s, manual button + hotkey)
4. Whisper live transcription (faster-whisper, live mic, en / ar / mixed)
5. Ollama summaries, transcript → summary view in the UI
   (+ model benchmark 2026-11-10: default summary model gemma4:e2b / qwen3:4b; re-tested on the demo laptop
   2026-11-10 → gemma4:e2b also on CPU-only 8 GB machines, see "Demo laptop session")
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


## Demo laptop session (2026-11-10) ✅ steps 1-6 · ⏳ Wi-Fi-off check, step 7 drills
Brief: docs/LAPTOP_SESSION.md (8 steps). Laptop-specific settings go in config/local.yaml (gitignored).

**Outcome (user, 2026-11-10: "satisfactory, the blink works well enough"; run this setup on this laptop and
similar limited-resource machines).** Demo configuration = what `auto` now picks on a CPU-only machine with
>= 7 GB RAM; this laptop's config/local.yaml only sets the mic (`audio.device: "USBAudio1.0"`) and
`events.gesture: triple_blink`:

| | Demo laptop (i7-1355U, 8 GB, no NVIDIA GPU) |
|---|---|
| Whisper | `small`, CPU int8, beam 1, `mixed`, **single-encode on** (`transcription.single_encode: auto`) |
| Summary LLM | **gemma4:e2b** (auto rule changed: no NVIDIA GPU + >= 7 GB RAM; was qwen3:4b below 12 GB) |
| Trigger | triple blink, rule on the learned blink signal (copied blink-default-muse model; auto = rule) |
| Measured live | text 6-10 s after speech; summary 20-28 s in the LLM, ~27-37 s after the window closes |
| Must for the demo | only Bayn + one light browser tab open (memory!); Muse on before `bayn run`; wait ~30 s after start |

Code changes from this session (all tested; 214 passed, 3 skipped = headset ×2 + real LLM):
- D27 setup: OLLAMA_MODELS set before installing Ollama, tray app stopped (scripts/bayn.ps1).
- `SingleEncodeWhisperEngine` (transcription/engine_single.py) + switch `transcription.single_encode`
  (auto = CPU only) + regression tool `python -m bayn.transcription.regress_single` (E9: compares at T=0).
- D30 memory: `bayn run` sets `OPENBLAS_NUM_THREADS=1` for its processes (~785 → ~90 MB committed each).
- Summary auto rule (summarization/models.py): gemma4:e2b with no NVIDIA GPU and >= 7 GB RAM.
- D28 benchmark report works on a machine without a GPU (`report --out docs/BENCHMARK_LAPTOP.md`).
- D29 system tests pin their gesture (a config/local.yaml no longer breaks them).

**Next, on the dev PC:**
1. `git pull`, then `.\.venv\Scripts\python.exe -m pytest` (expect 214+ passed; GPU paths unchanged).
2. GPU regression of the single-encode engine:
   `.\.venv\Scripts\python.exe -m bayn.transcription.regress_single --models large-v3-turbo --device cuda`
   (needs the corpus recordings in data/recordings). 0 decode mismatches → decide whether `auto` should also
   turn it on with CUDA (engine_single.py `_load`; on CUDA the encoder is cheap, so the gain may be small).
3. Optional: `--device cpu` there too (the laptop ran base / small / medium / turbo: 0 mismatches).
4. Still open on the laptop (next visit): Wi-Fi-off offline run; step 7 (`-m hardware` tests, the real-LLM test
   with Ollama running, offline-readiness / 10-min false-trigger / gesture latency / headset drop-out drills).

**Machine (first look, full scan in step 2):** Intel Core i7-1355U (2 P + 8 E cores, 12 threads, 15 W), 7.6 GB RAM,
Intel UHD integrated graphics (no NVIDIA → CPU-only path), Windows 11 Home 26200, one drive C: (328 GB free),
Intel Wireless Bluetooth (radio was off). Nothing was copied from the dev PC (no models, no blink sessions).

**Step 1 – setup ✅** `.\bayn setup` from a fresh clone: CPU-only deps (.venv 0.7 GB), Whisper small + tiny (0.5 GB),
Ollama 0.40.15 (winget, 3.1 GB program), auto summary model **qwen3:4b** (RAM < 12 GB), `doctor` "ready".
Found D27: the Ollama installer's tray app pulled the model into `~\.ollama\models` → moved to `models\ollama`,
setup fixed (OLLAMA_MODELS set before the install, tray app stopped after it). Bluetooth switched on by the user.
Ollama autostart disabled (Startup shortcut kept as models/Ollama-startup-shortcut.lnk.bak).
Blink data copied from the dev PC (6 sessions, 31 min + blink-default-muse.joblib): the sessions arrived as
`.npz.zip` in data/training/ → renamed to .npz (an .npz is already a zip) and moved to data/training/default-muse/.
Model loads live with the learned signal (TP9 -0.681, AF7 0.032, AF8 -0.136, TP10 -0.719, 341.2 µV); Auto = rule.
Candidate models for the benchmark (user: "download all the models that can feasibly run"): Whisper tiny, base,
small, medium, large-v3-turbo; LLMs qwen3:4b, gemma4:e2b, qwen2.5:3b, gemma3:1b (~10.6 GB more). Left out:
large-v3 (RTF 1.65 on 4 desktop threads, 3.1 GB RAM), gemma4:e4b (6.6 GB RAM), qwen2.5:7b / qwen3:8b (~5 GB, slower
and scored lower than gemma4:e2b).

**Step 2 – scan ✅** (details: TESTING.md §3) ASUS Vivobook 15 X1504VA. i7-1355U, 12 threads, 15 W. **7.6 GB RAM
in one 8 GB DDR4-3200 module (single channel)**: CPU LLM speed is limited by memory bandwidth, and only ~2.2 GB was
available with VS Code open, so gemma4:e2b (~4.6 GB) is the open question. Plugged in, "Best performance" power
mode on AC and battery, battery saver off. No NVIDIA GPU. Intel Bluetooth (only adapter) works, 6 BLE devices seen.
Audio: built-in Realtek mic array only; the wireless mic isn't connected yet. Paired BT: soundcore Q20i.
Startup apps that use RAM: OneDrive, Steam, Teams, Slack (close them for the demo).

**Step 3 – wireless mic ✅ (Muse coexistence ✅ in step 6: 6 min together, no audio warnings)** The demo mic is a **USB 2.4 GHz wireless lavalier** ("USBAudio1.0",
VID 4C4A / PID 4155), not Bluetooth: no hands-free profile, no load on the BLE adapter. Set in config/local.yaml
(`audio.device: "USBAudio1.0"`, matched by name). 30 s test through Bayn's MicSource (16 kHz, 0 dropped chunks):
speech -14 dBFS (90th pct), peak -1.8 dBFS on the loud sentence, **0 % clipped but under 2 dB headroom**; noise floor
-86 dBFS (the transmitter has a noise gate: 56 % of samples are digital silence), SNR ~72 dB; wideband but dark
(-25 dB at 3-4 kHz, energy up to ~7 kHz). Whisper small (CPU): read text recognised word for word.
To do: lower the input level a little for headroom; test with the Muse connected (both use 2.4 GHz).
Early warning: that small transcription took 30 s for 30 s of audio (RTF ~1.0, with model downloads running in the
background); the step 5 benchmark measures it properly.

**Step 4 – corpus recorded ✅** All 12 scripts with the wireless mic, demo-like (user, 2026-11-10, input level
lowered first). 29-37 s each, 0 % clipped, peaks -2.5 to -4 dBFS, speech ~-19 dBFS, noise floor -81 to -91 dBFS.
Recordings in data/recordings/ (not committed). Model downloads: Whisper base / medium / large-v3-turbo done;
the gemma4:e2b pull stalled at ~3.3 GB for 10 min (open connections, no progress) → stopped and restarted, it
resumes from the partial file.

**Step 5 – benchmark ✅ (user decisions, 2026-11-10: single-encode on for CPU, gemma4:e2b for 8 GB CPU machines)** All candidate models downloaded (gemma4:e2b, qwen2.5:3b, gemma3:1b pulled
and verified; Ollama models 9.15 GB). `benchmark stt --profiles cpu` (10 threads), 12 recordings, mean:

| Whisper | WER en | WER ar | WER mixed | mixed key terms | RTF | s / utterance |
|---|---|---|---|---|---|---|
| tiny | 12 % | 78 % | 78 % | 5 % | 0.11 | ~0.7 |
| base | 4 % | 65 % | 70 % | 10 % | 0.24 | ~1.6 |
| small (auto) | 5 % | 46 % | 53 % | 5 % | **0.59** | **~6** |
| medium | 26 %* | 30 % | 40 % | 20 % | 2.2 | ~16 |
| large-v3-turbo | 1 % | **16 %** | **30 %** | 29 % | 2.2 | ~24 |

\* mixed mode picked Arabic for half of two English clips and translated them (F2-type language-ID flip).
**Nothing accurate is real time on this laptop.** Diagnosis (scratchpad scripts, 2026-11-10):
- Cost is per utterance, not per second of speech: every utterance is padded to Whisper's 30 s window.
- **`mixed` runs the encoder twice per utterance** (detect_language, then transcribe); on this CPU the encoder is
  ~half the time (small: detect 2.85 s + transcribe 3.23 s per utterance; turbo 11.7 + 12.1 s).
- Threads don't help (4 = 6 = 10 = 12 threads, clock ~2.0-2.4 GHz): the 15 W chip is power-limited.
- No temperature-fallback re-decodes happened.
- Prototype "one encoder pass" (detect on exactly the frames transcribe encodes, reuse the encoder output):
  small RTF 0.59 → **0.37**, 6.2 → **3.9 s per utterance**, 11/12 transcripts identical; the 12th differs only
  because one utterance's language call is a coin flip (P(en) 0.67 vs 0.48). Proposed as a code change.
- User (2026-11-10): build the one-encoder-pass engine as a **separate copy** and regression-test it here and on
  the dev PC before it replaces anything → `transcription/engine_single.py` (`SingleEncodeWhisperEngine`, a
  subclass; engine.py untouched, not used live yet), `python -m bayn.transcription.regress_single [--models ...]
  [--device cpu|cuda]` (original vs copy on the corpus recordings; fails on any decode mismatch), 3 integration
  tests in tests/test_transcription_single.py (tiny): pass.
- MyASUS fan profile was already "Performance": ~2.0-2.4 GHz under load is this 15 W chip's limit.
- **LLM gate** (`benchmark mem` + 6 summaries per model of the incident scripts from the perfect text, CPU, ~3 GB
  RAM available with VS Code open): gemma4:e2b **1.7 GB resident** (dev PC 4.6 GB: its per-layer embedding tables
  are memory-mapped and only looked up, so Windows keeps few pages), **14-23 s** per summary, facts 43-71 %;
  qwen3:4b (the auto choice for < 12 GB RAM) 2.7 GB, 11-57 s (4/6 over 30 s), mostly 29-43 %; qwen2.5:3b 2.1 GB,
  17-35 s, 0-43 %; gemma3:1b 0.9 GB, 16-24 s, 29-57 %, one invented "unconscious". → full quality run narrowed to
  gemma4:e2b vs qwen3:4b on reference + cpu/small + cpu/base + cpu/large-v3-turbo. If it holds, the auto rule
  (gemma4:e2b only with RAM >= 12 GB) is wrong for this laptop.
- **Regression, original vs single-encode copy** (`regress_single --device cpu`, 37 utterances of the 12
  recordings): **0 decode mismatches on all four models.**

  | Whisper | identical | other | RTF | s / utterance | less time | mean WER |
  |---|---|---|---|---|---|---|
  | base | 35/37 | 2 sampling-only | 0.24 → 0.15 | 2.50 → 1.62 | 35 % | 46.1 → 46.1 % |
  | small | 36/37 | 1 language flip (en-incident utt 1, P(en) 0.67 → 0.52 ar) | **0.58 → 0.36** | 6.06 → 3.83 | 37 % | 34.4 → 35.9 % |
  | medium | 36/37 | 1 sampling-only | 1.62 → 1.20 | 17.0 → 12.6 | 26 % | 31.9 → 31.7 % |
  | large-v3-turbo | **37/37** | – | 2.10 → 1.10 | 22.1 → 11.6 | 48 % | 15.7 → 15.7 % |

  "Sampling-only" = texts differed but are identical when both engines decode at temperature 0: on garbled
  Arabic, faster-whisper's temperature fallback samples randomly and the original engine disagrees with itself
  (3 runs → 3 texts, E9). The first run counted those as mismatches; the tool and test now re-check at T=0.
  Pending: the same run on the dev PC (`--models large-v3-turbo --device cuda`).
- **LLM quality run (192 summaries, CPU): it holds.** Facts per transcript source, gemma4:e2b vs qwen3:4b:
  reference 75 vs 70 %, turbo 84 vs 57 %, small **70 vs 45 %**, base 60 vs 35 % (Arabic speech via small: 60 vs
  44 %, mixed: 65 vs 27 %). Speed over 96 summaries each: gemma4:e2b **19 s mean / 29 s slowest**, qwen3:4b 35 s /
  68 s. qwen3:4b also wrote empty summaries 19 times (~20 tokens, "Summary: / Unclear: none", always English
  output from a transcript). Invented content: 1/192 (gemma4:e2b wrote "سقطت" = fell, on ar-incident via turbo).
- Report: **docs/BENCHMARK_LAPTOP.md** (`benchmark report --out ...`; docs/BENCHMARK.md untouched). The report
  generator only used `gpu` LLM runs for quality (sections 2, 3, 5 were empty on a machine without a GPU) and
  took CPU speed from one arbitrary transcript source: fixed (`quality_profile`, `cpu_speed` in benchmark.py;
  dev-PC output unchanged, +1 integration test). Its section 5 picks **base + gemma4:e2b** for an 8 GB CPU
  machine, because small misses RTF ≤ 0.5 with the current engine (0.59); with the single-encode copy small is
  0.36 → small + gemma4:e2b (70 % facts vs 60 %).
- **config/local.yaml** now pins `transcription.model: small` (= auto) and `summary.model: gemma4:e2b`
  (overrides auto's qwen3:4b). → Adopted (user, 2026-11-10): auto picks gemma4:e2b with no NVIDIA GPU from
  7 GB RAM (`RAM_GB_FOR_LARGE_CPU_ONLY`; an "8 GB" machine reports ~7.6 GB); the pins were removed from local.yaml.

**Step 6 – end to end ✅ (Wi-Fi-off check ⏳)** Run 1 (2026-11-10, `bayn run --source sim`, wireless mic, Chrome dashboard,
VS Code + Claude Code open): everything starts (mic found by name, Whisper small CPU 5.5 s, gemma4:e2b loaded in
42 s, trigger = triple blink) and English speech is transcribed correctly, but **not in real time**: Whisper took
**8-34 s per utterance** live (benchmark: ~6 s), so lines arrived 12-35 s after speech and the queue grew.
Measured: transcriber 9.3 of 12 threads, Chrome ~1.3, everything else ~0; clock ~1.9 GHz on AC (as in the
benchmark); only ~1-2 GB RAM available (llama-server commits 5.3 GB, each Bayn service ~0.8 GB mostly paged
out), ~1,400 pages/s read from disk. Suspects: temperature-fallback re-decodes on conversational speech (5
temperatures × 5 samples; read corpus speech rarely triggers them) and memory pressure. Next: record 2 min of
natural speech, replay offline (default / T=0 only / single-encode / 4 threads). Stopped at the user's request
(looking for a lighter browser for the dashboard); RAM available after stopping: 3.2 GB.
- **Diagnosis** (2 min of the user's natural speech on the wireless mic, 13 utterances, median 10.4 s):
  offline, Whisper small takes **5.0 s per utterance** (RTF 0.54), same as the benchmark; fallback re-decodes
  are NOT the cause (1 in 13; greedy-only 5.4 s); single-encode **3.6 s** (RTF 0.39); 4 threads 6.6 s (slower).
  Replayed through the live pipeline (`--audio file`, gemma4:e2b loaded, no browser): 5-6 s per utterance,
  except one **22 s stall** when available RAM hit **12 MB** and Windows read up to 128,000 pages/s.
  → the live slowdown is **memory**: with Bayn off only 3.3 GB is available (Windows + VS Code + Claude Code),
  and Bayn needs about that at peak (gemma4:e2b 1.7 GB+, Whisper ~0.5 GB, services, browser).
- Each light Bayn service commits ~785 MB at import: numpy's and scipy's OpenBLAS reserve per-thread buffers
  (12 threads). `OPENBLAS_NUM_THREADS=1` → 77 MB (resident unchanged ~100 MB); system commit was 22 / 24.9 GB.
- **Fixes applied (user, 2026-11-10):** `bayn run` sets `OPENBLAS_NUM_THREADS=1` for its processes;
  `transcription.single_encode` switch (default false; on in this laptop's config/local.yaml). Re-run of the
  replay (run B, no browser): light services commit 51-103 MB each (was ~785), system commit 16.6 GB (was ~22);
  Whisper **2.7-3.7 s per utterance** once gemma4:e2b has loaded (was 5-6 s); slow ones (7-13 s) only during
  the model's 30 s load. Next: the user runs `bayn run` from a plain terminal with VS Code closed (demo conditions).
- **Demo-conditions run (user, 2026-11-10 21:13-21:20, `bayn run --no-browser` from a plain terminal, VS Code
  closed, real Muse-3EE1, wireless mic, dashboard in a separate browser) – user: "satisfactory, the blink works
  well enough".** From data/step6-demo.log and the 3 saved events:
  - Startup: Whisper small (single-encode) ready in 5.5 s, gemma4:e2b loaded in 17.3 s, ML blink model loaded
    (auto keeps the rule, as decided). Muse connected after ~80 s (3 × BOARD_NOT_READY retries while it came up);
    mains 60 Hz detected. AF8 quality "bad" all session (the known noisy AF8 sensor, D25).
  - Trigger: 2 triple blinks → 2 events (21:17:07, 21:18:40); 1 jaw-clench event while the user had switched
    the gesture to clench in the dashboard (no speech in it → "No speech was captured", no LLM call).
  - Transcript: Whisper 3.7-8.2 s per utterance (longer for continuous reading cut at 15 s, and Arabic);
    **6-10 s** from end of speech to text on screen. Correct English; Arabic mostly right with ASR slips
    (بساوة for مساواة, القائد الثهبية).
  - Summaries: English **19.7 s** in the LLM, 27 s after the window closed; Arabic **28.0 s**, 37 s after
    (transcript waited 9 s for the sentence in progress). Both accurate; the Arabic one repaired the ASR slips
    (مساواة, مضروب) and invented nothing; one odd word (يُجى). EEG section written by code, as designed.
  - Not reached: the 2-3 s live-line target (6-10 s here, accepted by the user for the demo).

## Seizure-detector socket with a mock model (2026-11-10) ✅ (user confirmed 2026-11-11)
User request (after the laptop demo): "a mock ML model that detects seizures and plug it into the socket",
trained on real public data with a disclaimer that it isn't validated on Muse data.
- Dataset search → **CHB-MIT** (PhysioNet, ODC-By 1.0, 256 Hz scalp EEG). Subset of 8 patients (1.9 GB in
  data/datasets/chbmit, gitignored): `python -m bayn.seizure.chbmit download`. Minimal EDF reader (no new dependency).
- Four Muse-like channels approximated from the bipolar montage; 14 features per 4 s window every 1 s, mostly relative
  to the person's own baseline (bayn.seizure.features, identical offline and live).
- Leave-one-patient-out: logistic regression **21/28 seizures, median 20 s, 3.6 false alarms/h** at threshold 0.9
  (random forest worse: 8/28). Shipped: src/bayn/seizure/assets/model.json (2 KB, coefficients, no pickle) +
  chbmit_clip.npz (a real seizure of chb23, not used for training, 249 KB). Model card: docs/SEIZURE_MODEL.md.
- `SeizureModelDetector` in the `Detector` socket: P >= 0.9 for 5 windows, 120 s refractory; event kind "seizure",
  red banner "mock model, not a diagnosis"; summary prompt calls it a mock research detector. Off by default
  (`events.seizure.enabled`), on in `bayn demo`; dashboard switch; simulator button replays the clip.
- Tests: 11 new (EDF, mapping, features, labels, alarms, model file, fires on the clip, ignores 12 gestures and
  artefacts, manager event, dashboard switch).

Next (user's list, 2026-11-10): 2) easier setup, with no manual copying of blink training; 3) easy testing and
demoing for whoever downloads the repo (judges); 4) repo and docs cleanup, GIFs, demo videos.

## Setup without blink training (2026-11-11) ✅ (user confirmed 2026-11-11)
User request: "make the setup easier (no need to manually move training for blink detection)".
- **Built-in Muse blink pattern** (src/bayn/events/assets/blink_muse.json: the 4 weights + threshold learned from the
  user's 6 sessions; no recordings shipped). A real headset uses it until the person calibrates or trains.
- **Calibrate… now learns the person's blink pattern** from its three triple blinks (about a minute), unless a
  trained model is loaded; saved in data/profiles and reloaded. Dashboard hint says which pattern is in use
  (trained / calibrated / built-in / forehead); the old "Train the ML detector" hint is gone (ML isn't needed).
- **Measured on the real sessions, held out** (`.\bayn blink calibration`): forehead signal 34 % caught, 2.6 false/min
  (what a fresh install did before, D31); Calibrate learning the pattern **66 %, 0.52/min**; 26 min of training
  67 %, 0.30/min. So training sessions are optional now; the laptop no longer needs data/training or data/models.
- Tests: tests/test_blink_setup.py (6).

**To test (headset):** delete or rename data/profiles/default-muse.json (or use a new profile), `.\bayn run`, set the
trigger to triple blink → the hint says "built-in Muse blink pattern" → try some triple blinks → **Calibrate…** →
the hint says "your calibrated blink pattern" → triple blinks again. (The trained model in data/models still wins
on the dev PC; rename it to test the fresh-install path.)

Next: 3) easy testing and demoing for judges; 4) repo and docs cleanup, GIFs, demo videos.

## Showcase + recorded preview for judges (2026-11-11) ✅ (user confirmed 2026-11-11)
User request: "streamline it so testing and demoing is easy for whoever downloads the repo (judges)" and "a way for
judges to preview as much as possible"; GIFs before any videos. Speech: the user's wireless-mic corpus recordings
(user: "those can be public"; the 4 used are in samples/showcase/, all 12 stay in data/recordings/wireless).
- **`.\bayn showcase`** (also `make showcase`): no headset, no mic. 4 scenes in 212 s, looping: jaw clench + Arabic,
  button + English, triple blink + mixed, mock seizure model + English lesson. Caption banner in the dashboard; the
  trigger setting follows the scene. Script: samples/showcase/showcase.json; driver: src/bayn/showcase.py.
- **Recorded preview**: `python -m bayn.preview record` (during a showcase, one loop) → `export` → docs/preview/
  (3.8 MB; the real dashboard code replaying the recording; open index.html from disk or host it) → `gif` →
  docs/media/*.gif (headless Edge + Pillow, reproducible; no triple-blink GIF: the user records one on the real
  headset with a camera, docs/media/triple-blink.gif). README: new "Try it (judges)" section with the GIFs.
- Fixes found by the showcase: D32 (seizure alarm swallowed by a gesture cooldown), D33 (gesture change reset the
  seizure monitor). Also: EEG over the WebSocket was float32 noise digits (4x the JSON), now rounded properly;
  simulator markers on the plot no longer overlap event labels.
- Tests: tests/test_showcase.py (6) + 2 seizure regressions → 239 passed, 3 skipped (~3.5 min).

**To test:** `.\bayn showcase` (watch one loop, ~4 min incl. the last summary); open docs/preview/index.html in a browser
(no server); look at the README's GIFs. On the laptop (CPU) the summaries come later (~30 s), the scenes are spaced
for that.

Open: host the preview (GitHub Pages: Settings → Pages → main, /docs → https://omar0rawas.github.io/bayn/preview/)?
Next: 4) repo and docs cleanup, demo videos.

## Repo + docs cleanup (2026-11-11) ✅ (user confirmed 2026-11-11)
User request: scan and clean up the repo, clean up the "about", keep the warnings, say that absence seizures are the
target but the triggers are deliberate for development, mention the seizure model and its data source, list the
libraries and models explicitly; then a scan and a self-rating.
- README restructured: **About** (absence seizures as the target; deliberate triggers for development; the detector
  socket; the mock CHB-MIT model, not an absence-seizure detector), Try it (judges), How it works, **Built with**
  (libraries by area, models with what they're used for and where they run, data source), Quickstart, Triggers,
  seizure socket, transcription, summaries, laptop, configuration, Muse notes, layout, **Documentation** index (replaces
  the stale phase table), testing, privacy, **Acknowledgements** (CHB-MIT citation as ODC-By requires).
- GitHub About: new description + 12 topics. pyproject description updated.
- Model card (and its generator): Bayn targets absence seizures; CHB-MIT doesn't label seizure types, so the mock
  model is not an absence-seizure detector.
- Removed: the inert `SeizureDetectorStub` (superseded by the mock model, which now runs the detector contract tests;
  the plug-in guide moved into seizure_model.py), the placeholder docs/img/dashboard.svg.
- Fixed: D34 (backspace characters in 4 docs), D35 (`httpx` missing from the runtime dependencies); stale texts (training
  dialog said a classifier learns; training.py said double blink).

