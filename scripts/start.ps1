#requires -Version 7.0
[CmdletBinding()]
param([switch]$NoBuild, [switch]$SkipSample)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$pythonCommand = (Get-Command python -ErrorAction Stop).Source
Push-Location -LiteralPath $projectDirectory
try {
    & docker compose config --quiet
    if ($LASTEXITCODE -ne 0) { throw 'Docker Compose configuration is invalid.' }
    $composeArguments = @('compose', 'up', '-d')
    if (-not $NoBuild) { $composeArguments += '--build' }
    & docker @composeArguments
    if ($LASTEXITCODE -ne 0) { throw 'Docker Compose could not start. Check Docker Desktop and access to its Linux engine.' }
    & $pythonCommand -m local_voice.cli wait --timeout 1800
    if ($LASTEXITCODE -ne 0) {
        & docker compose logs --tail 80 tts
        throw 'The GPU speech engine failed its readiness check.'
    }
    & (Join-Path $PSScriptRoot 'configure-mcp.ps1')
    if (-not $SkipSample) {
        & $pythonCommand -m local_voice.cli benchmark --runs 3 --play --require-interactive
        if ($LASTEXITCODE -ne 0) { throw 'Streaming playback or its latency/throughput target failed. Inspect the report in .runtime/streaming-benchmark.json.' }
    }
    Write-Host 'Ready. Reload local-qwen-voice in Claude/Codex/ChatGPT Desktop so the running bridge uses streaming playback.'
}
finally { Pop-Location }
