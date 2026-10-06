#requires -Version 7.0
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$pythonCommand = (Get-Command python -ErrorAction Stop).Source
$configurationPath = Join-Path $projectDirectory '.mcp.json'
$configuration = @{}
if (Test-Path -LiteralPath $configurationPath) {
    $configuration = Get-Content -Raw -LiteralPath $configurationPath | ConvertFrom-Json -AsHashtable
}
if (-not $configuration.ContainsKey('mcpServers')) { $configuration['mcpServers'] = @{} }
$configuration['mcpServers']['local-qwen-voice'] = @{
    type = 'stdio'
    command = $pythonCommand
    args = @((Join-Path $projectDirectory 'mcp_server.py'))
}
$configuration | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $configurationPath -Encoding utf8NoBOM
Write-Host "Configured local-qwen-voice in $configurationPath"

