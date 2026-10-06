#requires -Version 7.0
[CmdletBinding()]
param([switch]$Play, [switch]$UnitOnly)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectDirectory
try {
    & python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Unit and MCP protocol tests failed.' }
    & docker compose config --quiet
    if ($LASTEXITCODE -ne 0) { throw 'Docker Compose configuration is invalid.' }
    if (-not $UnitOnly) {
        & docker compose exec -T tts nvidia-smi
        if ($LASTEXITCODE -ne 0) { throw 'Cannot access the GPU inside the Compose container.' }
        & docker compose exec -T tts python -m pip check
        if ($LASTEXITCODE -ne 0) { throw 'Container dependency check failed.' }
        $smokeArguments = @('-m', 'local_voice.cli', 'benchmark', '--runs', '3', '--require-interactive')
        if ($Play) { $smokeArguments += '--play' }
        & python @smokeArguments
        if ($LASTEXITCODE -ne 0) { throw 'Real GPU speech synthesis failed.' }
    }
}
finally { Pop-Location }
