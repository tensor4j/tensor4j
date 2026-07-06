# Enhanced interactive chat — quality sampling, longer replies, focus system prompt.
# Model may riff, ask follow-up questions, or wander (see chat-enhanced.sh on Git Bash/WSL).
#
# Usage:
#   .\chat-enhanced.ps1
#   .\chat-enhanced.ps1 -Build

$ErrorActionPreference = "Stop"

if (-not $env:TENSOR4J_CHAT_MODE) { $env:TENSOR4J_CHAT_MODE = "quality" }
if (-not $env:TENSOR4J_CHAT_MAX_TOKENS) { $env:TENSOR4J_CHAT_MAX_TOKENS = "256" }
if (-not $env:TENSOR4J_CHAT_MIN_TOKENS) { $env:TENSOR4J_CHAT_MIN_TOKENS = "2" }
if (-not $env:TENSOR4J_CHAT_SYSTEM_PROMPT) { $env:TENSOR4J_CHAT_SYSTEM_PROMPT = "focus" }
if (-not $env:TENSOR4J_CHAT_DEBUG) { $env:TENSOR4J_CHAT_DEBUG = "true" }

& (Join-Path $PSScriptRoot "chat.ps1") @args
exit $LASTEXITCODE
