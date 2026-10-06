#requires -Version 7.0
# Optional: expose the same bridge to Claude Code in every project.
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'register-global.ps1') -Client Claude
