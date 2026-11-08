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
    & $Py -m bayn.doctor
  }
  "demo"   { Require-Venv; & $Py -m bayn.run --demo @Rest }
  "run"    { Require-Venv; & $Py -m bayn.run @Rest }
  "check"  { Require-Venv; & $Py -m bayn.eeg.check @Rest }
  "models" { Require-Venv; & $Py -m bayn.transcription.models @Rest }
  "mics"   { Require-Venv; & $Py -m bayn.transcription.audio --list-devices }
  "record" { Require-Venv; & $Py -m bayn.transcription.record @Rest }
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
  models   download Whisper models (models --all: GPU + CPU models, e.g. to copy to a laptop)
  mics     list microphones          sample-audio  regenerate the demo/test WAVs (Windows TTS)
  record   record the mic to data/recordings/<name>.wav   (record classroom_mixed --seconds 45)
  eval-stt score transcription strategies on a recording  (eval-stt --scenario mixed)
  bridge   EEG bridge only           ui   dashboard only
  test     run the test suite        doctor  check this machine
  clean    remove caches
"@
  }
}
exit $LASTEXITCODE
