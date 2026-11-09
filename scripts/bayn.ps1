<#
.SYNOPSIS
  Bayn task runner for Windows (mirrors the Makefile).

.EXAMPLE
  .\bayn setup      # create .venv and install pinned deps
  .\bayn demo       # simulated EEG + scripted gestures, opens the dashboard
  .\bayn test

  Call it through ..\bayn.cmd: Windows blocks .ps1 scripts by default, and the
  wrapper bypasses that for the single call without changing system settings.
#>
param(
  [Parameter(Position = 0)][string]$Task = "help",
  [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"

function Find-BasePython {
  # Prefer the py launcher (picks newest 3.12+), fall back to python on PATH.
  # (try/catch: PowerShell 5.1 turns a native command's stderr into an error under "Stop")
  foreach ($v in "3.13", "3.12") {
    if (Get-Command py -ErrorAction SilentlyContinue) {
      try { & py "-$v" -c "import sys" 2>$null } catch { }
      if ($LASTEXITCODE -eq 0) { return @("py", "-$v") }
    }
  }
  if (Get-Command python -ErrorAction SilentlyContinue) {
    & python -c "import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)"
    if ($LASTEXITCODE -eq 0) { return @("python") }
  }
  throw "Python 3.12+ not found. Install it from https://www.python.org/downloads/ (tick 'Add to PATH')."
}

function Require-Venv {
  if (-not (Test-Path $Py)) { throw "No .venv yet - run: .\bayn setup" }
}

switch ($Task) {
  "setup" {
    $base = Find-BasePython
    if (-not (Test-Path $Py)) {
      Write-Host "Creating .venv with $($base -join ' ')..."
      if ($base.Count -gt 1) { & $base[0] $base[1] -m venv .venv } else { & $base[0] -m venv .venv }
    }
    & $Py -m pip install --upgrade pip --quiet
    & $Py -m pip install -r requirements\dev.txt --quiet
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
      Write-Host "NVIDIA GPU found - installing CUDA libraries for Whisper (~1 GB)..."
      & $Py -m pip install -r requirements\gpu.txt --quiet
    }
    & $Py -m pip install -e . --no-deps --quiet
    if (-not (Test-Path .env)) { Copy-Item .env.example .env }
    Write-Host "Downloading Whisper models (one time)..."
    & $Py -m bayn.transcription.models
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue) -and -not (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe")) {
      if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Installing Ollama (local LLM runtime)..."
        winget install --id Ollama.Ollama -e --silent --accept-source-agreements --accept-package-agreements
      } else { Write-Host "Install Ollama from https://ollama.com, then run: .\bayn models" }
    }
    # Keep Ollama's models next to the Whisper models (in this folder), not on a possibly small C: drive,
    # unless the user already chose a location. Also keep Ollama local-only (no ollama.com cloud features).
    if (-not [Environment]::GetEnvironmentVariable("OLLAMA_MODELS", "User")) {
      $om = Join-Path $Root "models\ollama"
      [Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $om, "User"); $env:OLLAMA_MODELS = $om
      Write-Host "Ollama models will be stored in $om"
    }
    [Environment]::SetEnvironmentVariable("OLLAMA_NO_CLOUD", "1", "User"); $env:OLLAMA_NO_CLOUD = "1"
    Write-Host "Downloading the summary model (one time)..."
    & $Py -m bayn.summarization.models
    & $Py -m bayn.doctor
  }
  "demo"   { Require-Venv; & $Py -m bayn.run --demo @Rest }
  "run"    { Require-Venv; & $Py -m bayn.run @Rest }
  "check"  { Require-Venv; & $Py -m bayn.eeg.check @Rest }
  "models" { Require-Venv; & $Py -m bayn.transcription.models @Rest; & $Py -m bayn.summarization.models @Rest }
  "eval-summary" { Require-Venv; & $Py -m bayn.summarization.evaluate @Rest }
  "mics"   { Require-Venv; & $Py -m bayn.transcription.audio --list-devices }
  "record" { Require-Venv; & $Py -m bayn.transcription.record @Rest }
  "corpus" { Require-Venv; & $Py -m bayn.corpus @Rest }
  "eval-stt" { Require-Venv; & $Py -m bayn.transcription.evaluate @Rest }
  "sample-audio" { Require-Venv; & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "make_sample_audio.ps1") }
  "bridge" { Require-Venv; & $Py -m bayn.eeg.bridge @Rest }
  "ui"     { Require-Venv; & $Py -m bayn.ui.server @Rest }
  "test"   { Require-Venv; & $Py -m pytest @Rest }
  "doctor" { Require-Venv; & $Py -m bayn.doctor }
  "clean"  {
    Get-ChildItem -Recurse -Directory -Include __pycache__, .pytest_cache, *.egg-info -ErrorAction SilentlyContinue |
      Remove-Item -Recurse -Force
    Write-Host "Cleaned caches (data/ and .venv kept)."
  }
  default {
    Write-Host @"
Bayn tasks:
  setup    create .venv, install pinned deps (+ CUDA libs if NVIDIA GPU), download models, run doctor
  demo     simulated EEG + scripted gestures + live microphone + dashboard (no headset)
  run      real Muse 2 (default) + microphone + dashboard
           options: --source lsl|sim   --audio file (play samples/audio/demo.wav instead of the mic)
  check    connect to the headset for 10 s and report signal quality (check --name Muse-3EE1)
  models   download Whisper + summary models (models --all: every variant, e.g. before moving to a laptop)
  eval-summary  score summary models on the classroom scenarios (eval-summary --models qwen2.5:3b qwen2.5:7b)
  mics     list microphones          sample-audio  regenerate the demo/test WAVs (Windows TTS)
  corpus   list the recording corpus (docs/CORPUS.md) and what is recorded; corpus show mx-pi opens one
  record   record a corpus script (opens it in big type, Enter stops): record mx-pi
  eval-stt score transcription on recordings: eval-stt --corpus all [--models small large-v3-turbo] [--device cpu]
  bridge   EEG bridge only           ui   dashboard only
  test     run the test suite        doctor  check this machine
  clean    remove caches
"@
  }
}
exit $LASTEXITCODE
