<#
.SYNOPSIS
  Move Claude Code session history for this repo between machines (e.g. dev PC -> demo laptop).

.DESCRIPTION
  Claude Code keeps sessions + memory in  %USERPROFILE%\.claude\projects\<encoded repo path>\
  where every non-alphanumeric character of the repo path becomes '-'. The folder name
  therefore changes when the repo lives at a different path on the other machine;
  'import' recreates it under the name that matches THIS machine's checkout.

  The zip contains your full conversation history. It is gitignored - do not commit or share it.
  The repo itself also carries the essentials in CLAUDE.md and docs/PROGRESS.md.

.EXAMPLE
  .\scripts\claude-session.ps1 export                     # on the PC  -> .\claude-session-export.zip
  .\scripts\claude-session.ps1 import .\claude-session-export.zip   # on the laptop, inside the cloned repo
  claude --resume                                         # pick the session to continue
#>
param(
  [Parameter(Mandatory = $true, Position = 0)][ValidateSet("export", "import")][string]$Action,
  [Parameter(Position = 1)][string]$Zip = ".\claude-session-export.zip"
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Split-Path -Parent $PSScriptRoot)).Path.TrimEnd('\')
$Projects = Join-Path $env:USERPROFILE ".claude\projects"
$Encoded = ($Root -replace '[^A-Za-z0-9]', '-')

if ($Action -eq "export") {
  # Windows paths are case-insensitive, so match the folder name the same way.
  $src = Get-ChildItem $Projects -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -ieq $Encoded } | Select-Object -First 1
  if (-not $src) { throw "No Claude session folder for this repo (looked for $Projects\$Encoded)." }
  if (Test-Path $Zip) { Remove-Item $Zip -Force }
  Compress-Archive -Path (Join-Path $src.FullName "*") -DestinationPath $Zip
  Write-Host "Exported $($src.Name) -> $Zip"
  Write-Host "Copy it to the other machine (USB/cloud drive), clone the repo there, then run: .\scripts\claude-session.ps1 import <zip>"
}
else {
  if (-not (Test-Path $Zip)) { throw "Zip not found: $Zip" }
  $dest = Join-Path $Projects $Encoded
  New-Item -ItemType Directory -Force $dest | Out-Null
  # Never overwrite sessions that already exist on this machine.
  $tmp = Join-Path $env:TEMP ("claude-import-" + [guid]::NewGuid())
  Expand-Archive -Path $Zip -DestinationPath $tmp
  Get-ChildItem $tmp -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring($tmp.Length + 1)
    $target = Join-Path $dest $rel
    if (Test-Path $target) { Write-Host "skip (exists): $rel" ; return }
    New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
    Copy-Item $_.FullName $target
  }
  Remove-Item $tmp -Recurse -Force
  Write-Host "Imported into $dest"
  Write-Host "Now run 'claude --resume' from $Root and pick the session."
}
