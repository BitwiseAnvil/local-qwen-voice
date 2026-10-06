#requires -Version 7.0
[CmdletBinding()]
param(
    [ValidateSet('All', 'Codex', 'Claude')]
    [string]$Client = 'All',
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$entryPoint = Join-Path $projectDirectory 'mcp_server.py'
$pythonCommand = (Get-Command python -ErrorAction Stop).Source
$windowsProfilePath = [Environment]::GetFolderPath('UserProfile')
$windowsAccount = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$wrongAccount = $windowsAccount -match '(?i)\\CodexSandbox' -or (
    $env:USERPROFILE -and $windowsProfilePath -ine $env:USERPROFILE
)

# Native Codex resolves the Windows account profile, which can differ from
# USERPROFILE inside an agent sandbox. Never register into that sandbox account.
if ($wrongAccount) {
    $message = "This process runs as $windowsAccount ($windowsProfilePath), not your normal Windows account. Run this script from a normal PowerShell terminal."
    if (-not $CheckOnly) { throw $message }
    Write-Warning $message
}

$codexCommand = $null
$claudeCommand = $null
if ($Client -in @('All', 'Codex')) {
    $codexCommand = (Get-Command codex -ErrorAction Stop).Source
}
if ($Client -in @('All', 'Claude')) {
    $claudeCommand = (Get-Command claude -ErrorAction Stop).Source
}
if (-not (Test-Path -LiteralPath $entryPoint -PathType Leaf)) {
    throw "MCP entry point was not found: $entryPoint"
}

Write-Host "MCP server: local-qwen-voice"
Write-Host "Python: $pythonCommand"
Write-Host "Entry point: $entryPoint"
Write-Host "Clients: $Client (Codex registration also covers ChatGPT Desktop on the same host)"
if ($CheckOnly) {
    Write-Host 'Check only; no global configurations changed.'
    return
}

if ($codexCommand) {
    & $codexCommand mcp add local-qwen-voice -- $pythonCommand $entryPoint
    if ($LASTEXITCODE -ne 0) { throw 'Codex global registration failed; see the CLI error above.' }
    $registered = & $codexCommand mcp get local-qwen-voice --json
    if ($LASTEXITCODE -ne 0) { throw 'Codex could not read back the registered server.' }
    $server = ($registered -join "`n") | ConvertFrom-Json
    if ($server.transport.command -ine $pythonCommand -or
        @($server.transport.args).Count -ne 1 -or
        $server.transport.args[0] -ine $entryPoint) {
        throw 'Codex returned an unexpected server command after registration.'
    }
    Write-Host 'Registered globally for Codex CLI and ChatGPT Desktop on this host.'
}

if ($claudeCommand) {
    & $claudeCommand mcp add --scope user --transport stdio local-qwen-voice -- $pythonCommand $entryPoint
    if ($LASTEXITCODE -ne 0) {
        throw 'Claude global registration failed; see the CLI error above. If this exact server was already registered, check it with: claude mcp get local-qwen-voice'
    }
    Write-Host 'Registered globally for Claude Code CLI.'
}

Write-Host 'Restart the clients or reload MCP servers. Start Docker Compose separately for speech synthesis.'
