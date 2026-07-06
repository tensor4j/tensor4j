#!/usr/bin/env bash
# Interactive chat with "enhanced" sampling — quality mode, longer replies, focus system prompt.
# The model may riff, ask follow-up questions, or wander (interesting but not parity-tested).
#
# Same flags as chat.sh; only defaults differ. Override any TENSOR4J_* env var as usual.
#
# Usage:
#   ./chat-enhanced.sh
#   ./chat-enhanced.sh --build

set -euo pipefail

export TENSOR4J_CHAT_MODE="${TENSOR4J_CHAT_MODE:-quality}"
export TENSOR4J_CHAT_MAX_TOKENS="${TENSOR4J_CHAT_MAX_TOKENS:-256}"
export TENSOR4J_CHAT_MIN_TOKENS="${TENSOR4J_CHAT_MIN_TOKENS:-2}"
export TENSOR4J_CHAT_SYSTEM_PROMPT="${TENSOR4J_CHAT_SYSTEM_PROMPT:-focus}"
export TENSOR4J_CHAT_DEBUG="${TENSOR4J_CHAT_DEBUG:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/chat.sh" "$@"
