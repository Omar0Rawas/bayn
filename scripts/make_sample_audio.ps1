<#
.SYNOPSIS
  Generate the sample audio used by the file audio source and the tests, with Windows' built-in
  offline text-to-speech (English + Arabic). No internet needed.

.DESCRIPTION
  Reads the sentences from samples/audio/demo_script.json and writes:
    samples/audio/demo.wav       ~1 min mixed English/Arabic narration
    tests/data/en_short.wav      one English sentence (integration tests)
    tests/data/ar_short.wav      one Arabic sentence (integration tests)
  The Arabic voice (e.g. "Naayf") is a Windows "OneCore" voice. If missing: Settings >
  Time & Language > Speech > Add voices > Arabic. Other OSes: record your own clips.

.EXAMPLE
  .\bayn sample-audio
#>
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Tmp = Join-Path $env:TEMP ("bayn-tts-" + [guid]::NewGuid())
New-Item -ItemType Directory -Force $Tmp | Out-Null
# UTF-8 read is explicit: Windows PowerShell 5.1 would otherwise garble the Arabic text
$Script = Get-Content -Raw -Encoding UTF8 (Join-Path $Root "samples\audio\demo_script.json") | ConvertFrom-Json

# --- WinRT plumbing (PowerShell 5.1 has no 'await')
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, [Type]$type) {
  $t = $asTaskGeneric.MakeGenericMethod($type).Invoke($null, @($op)); $t.Wait(); $t.Result
}

$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$voices = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices
function Get-Voice([string]$lang) {
  $v = $voices | Where-Object { $_.Language -like "$lang*" } | Select-Object -First 1
  if (-not $v) { throw "No '$lang' text-to-speech voice installed (Settings > Time & Language > Speech > Add voices)." }
  $v
}

function Speak($line, [string]$outFile) {
  $synth.Voice = Get-Voice $line.lang
  $stream = Await ($synth.SynthesizeTextToStreamAsync($line.text)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
  $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
  $n = [uint32]$stream.Size
  $null = Await ($reader.LoadAsync($n)) ([uint32])
  $bytes = New-Object byte[] $n
  $reader.ReadBytes($bytes)
  [IO.File]::WriteAllBytes($outFile, $bytes)
}

$parts = @()
$i = 0
foreach ($line in $Script.demo) {
  $f = Join-Path $Tmp ("line{0:D2}.wav" -f $i); $i++
  Speak $line $f
  $parts += $f
}
$enShort = Join-Path $Tmp "en_short.wav"; Speak $Script.en_short $enShort
$arShort = Join-Path $Tmp "ar_short.wav"; Speak $Script.ar_short $arShort

# --- stitch with pauses, convert to 16 kHz mono 16-bit (python helper in the repo)
New-Item -ItemType Directory -Force (Join-Path $Root "tests\data") | Out-Null
& $Py -m bayn.transcription.audio_sources stitch (Join-Path $Root "samples\audio\demo.wav") @parts
& $Py -m bayn.transcription.audio_sources stitch (Join-Path $Root "tests\data\en_short.wav") $enShort
& $Py -m bayn.transcription.audio_sources stitch (Join-Path $Root "tests\data\ar_short.wav") $arShort
Remove-Item $Tmp -Recurse -Force
Write-Host "Wrote samples/audio/demo.wav, tests/data/en_short.wav, tests/data/ar_short.wav"
