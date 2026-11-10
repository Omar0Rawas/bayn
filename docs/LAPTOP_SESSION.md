# Demo-laptop session: brief for Claude Code

This brief is for the Claude Code session on the **demo laptop**. The user pastes the prompt below into a fresh session after cloning the repo. The dev-PC sessions wrote it on 2026-11-10, after Phase 6.

> **Done 2026-11-10 (steps 1-6; user: "satisfactory").** Results, the demo configuration, the code changes and what is still open (Wi-Fi-off check, step 7 drills, the GPU regression on the dev PC): docs/PROGRESS.md "Demo laptop session"; benchmark: docs/BENCHMARK_LAPTOP.md. For a later laptop visit, run only what that section lists as open.

## Before the session (done by the user)

1. Install **Git** and **Python 3.12+**: `winget install Git.Git Python.Python.3.12` (or python.org; tick "Add to PATH").
2. Install the **Claude Code** extension in VS Code and sign in.
3. Clone the repo onto the drive with the most free space (not a small C:):
   `git clone https://github.com/Omar0Rawas/bayn.git`, then open the folder in VS Code.
4. Optional, saves a lot of downloading: from the dev PC, copy onto a USB drive:
   - `models/whisper/`: at least `small` and `large-v3-turbo`, plus `tiny` for the tests;
   - `models/ollama/`: `gemma4:e2b` and `qwen3:4b`;
   - `data/training/` and `data/models/`: the user's labelled blink sessions and blink model.

   Copy each folder to the same place inside the cloned repo.
5. Have the wireless microphone, the Muse 2 headset (charged) and the laptop charger at hand.

## The prompt

```text
You are continuing the Bayn project on the DEMO LAPTOP (the hackathon demo machine). Earlier
sessions on the dev PC built everything. Read CLAUDE.md first; it holds the rules and the working
agreement. Then read docs/PROGRESS.md (end), docs/TESTING.md (Phase 6, model benchmark, Phases 7-8),
docs/BENCHMARK.md and docs/BLINK_DECISION.md, and this file (docs/LAPTOP_SESSION.md).

Goal of this session: make Bayn run well on THIS laptop with the WIRELESS microphone that will be used
in the live demo, choose the best models that stay real time here, and report back.

Rules that matter most here:
- Never hardcode hardware; auto-detection must keep working on other machines. Put laptop-specific
  choices in config/local.yaml (gitignored), and change defaults in code only if the data shows a
  default is wrong.
- Keep big files off a small C: drive. Models live in the repo's models/ folder (OLLAMA_MODELS ->
  models/ollama). Check free space BEFORE every download and tell me the size first.
- 100 % offline after setup. No cloud APIs, no Docker. PowerShell scripts stay ASCII (PowerShell 5.1).
- Don't run anything that needs the Muse headset until I say it's on and connected.
- I read the corpus scripts aloud myself; tell me exactly when and what to record.
- Stop at each numbered step below that needs me, and don't commit or push without asking.
- Don't change the blink decision criteria or the pre-registered results. Re-run reports, don't
  edit them.

Steps:
1. Install and set up: `.\bayn setup` (venv + pinned deps, CUDA libraries only if an NVIDIA GPU is
   present, Whisper models, Ollama via winget, summary model). If I copied models/ or data/ from the
   dev PC, reuse them instead of downloading. Then `.\bayn doctor` and fix everything it flags.
2. Scan the laptop and record its capabilities in docs (PROGRESS.md laptop section, TESTING.md
   test environment): CPU model, cores/threads, RAM, GPU and VRAM (or none), free disk per drive,
   Windows power plan (plugged in? battery saver throttles the CPU: check and tell me), Bluetooth
   adapter(s) (the Muse needs BLE; only one adapter may be active), audio devices.
3. Set up the wireless microphone: `.\bayn mics`, pick it, and set `audio.device` in
   config/local.yaml. Check level, clipping and noise with a short test recording. If it's a
   Bluetooth mic, note whether Windows uses the low-quality "hands-free" profile (16 kHz narrowband)
   and whether it interferes with the Muse's Bluetooth. Test both together once I connect the Muse.
4. Record the test corpus with that setup, demo-like (same distance, same room if possible):
   `.\bayn record <id>` for all 12 scripts in docs/CORPUS.md (`.\bayn corpus` shows what's done).
   Recordings stay local in data/recordings/ (never committed).
5. Compare models and pick the best that is still real time ON THIS LAPTOP:
   - `.\bayn benchmark stt --models tiny base small medium large-v3-turbo --profiles cpu`
     (add `gpu` if there is an NVIDIA GPU; skip large models if they clearly can't keep up);
   - `.\bayn benchmark llm --profile cpu --sources all --models gemma4:e2b qwen3:4b`
     (gpu profile if NVIDIA; only pull models that fit the disk and RAM/VRAM);
   - `.\bayn benchmark mem --profiles cpu --models gemma4:e2b qwen3:4b` (or gpu);
   - `.\bayn benchmark report --out docs/BENCHMARK_LAPTOP.md` (do NOT overwrite docs/BENCHMARK.md,
     which holds the dev-PC results).
   "Real time" here means: Whisper real-time factor <= 0.5 with all cores while the rest of Bayn
   runs; live transcript lines appear within ~2-3 s of the speaker pausing; a summary is on screen
   within ~30 s of an event window closing. Prefer accuracy on Arabic and mixed speech (English key
   terms inside Arabic) within those limits. Explain the trade-off and set the choice in
   config/local.yaml. If the auto rules pick wrongly for a laptop like this, propose a code change.
6. Check it's satisfactory, end to end, the way the demo will run: `.\bayn run` with the wireless mic
   (and the Muse once I say it's connected): live transcript latency, summary time and quality in
   English, Arabic and mixed, and the triple-blink trigger (with the copied blink model, or record one
   standard session via Blink detector -> Train..., then `.\bayn blink decide`). Then repeat with
   Wi-Fi off (offline check).
7. Run the tests, including the ones not yet run on this machine: `.\bayn test`; the real-LLM test
   with Ollama running; the hardware tests with the Muse ($env:BAYN_HARDWARE=1, `-m hardware`); and
   the pending items in docs/TESTING.md (anything marked as not done, especially Phases 7-8):
   - offline readiness;
   - the 10-minute normal-activity false-trigger test;
   - gesture-to-event latency;
   - headset drop-out and reconnect;
   - failure drills (headset off, mic unplugged, Ollama stopped);
   - README quickstart followed literally.
   Log defects (D#) and environment issues (E#) in TESTING.md.
8. Report back in a short summary:
   - the laptop's capabilities;
   - the chosen Whisper + summary models with their numbers;
   - mic findings;
   - what passed and what failed;
   - anything that risks the live demo, with recommendations.

   Update docs/PROGRESS.md and docs/TESTING.md, then ask me before committing and pushing, so the
   dev-PC sessions can pick the results up from GitHub.
```
