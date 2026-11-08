<#
.SYNOPSIS
  Bayn task runner for Windows (mirrors the Makefile).

.EXAMPLE
  .\scripts\bayn.ps1 setup      # create .venv and install pinned deps
  .\scripts\bayn.ps1 demo       # simulated EEG + scripted gestures, opens the dashboard
  .\scripts\bayn.ps1 test

  If scripts are blocked:  powershell -ExecutionPolicy Bypass -File .\scripts\bayn.ps1 setup
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
  if (-not (Test-Path $Py)) { throw "No .venv yet - run: .\scripts\bayn.ps1 setup" }
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
    & $Py -m pip install -e . --no-deps --quiet
    if (-not (Test-Path .env)) { Copy-Item .env.example .env }
    & $Py -m bayn.doctor
  }
  "demo"   { Require-Venv; & $Py -m bayn.run --demo @Rest }
  "run"    { Require-Venv; & $Py -m bayn.run @Rest }
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
  setup    create .venv, install pinned dependencies, run doctor
  demo     simulated EEG with scripted gestures + dashboard (no hardware)
  run      run with the configured EEG source  (e.g. run --source sim)
  bridge   EEG bridge only           ui   dashboard only
  test     run the test suite        doctor  check this machine
  clean    remove caches
"@
  }
}
exit $LASTEXITCODE
