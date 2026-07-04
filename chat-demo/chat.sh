#!/usr/bin/env bash
# Interactive chat with a local Llama 3.2 GGUF via tensor4j.
# Saves each session to ~/.local/conversations/<timestamp>-llama32-chat.md (readable)
# and <timestamp>-llama32-chat-audit.log (token audit) — updated after every turn.
#
# Usage:
#   ./chat.sh
#   ./chat.sh --gguf /path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf
#   TENSOR4J_CHAT_MAX_TOKENS=256 ./chat.sh
#
# Options (also available as env vars):
#   --gguf PATH          TENSOR4J_GGUF_PATH
#   --template NAME      TENSOR4J_CHAT_TEMPLATE (llama3|plain, default llama3)
#   --mode MODE          TENSOR4J_CHAT_MODE (quality|greedy)
#   --max-tokens N       TENSOR4J_CHAT_MAX_TOKENS
#   --save-dir PATH      TENSOR4J_CHAT_SAVE_DIR (default ~/.local/conversations)
#   --skip-build         do not run mvn install
#   -h, --help           show help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSOR4J_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SKIP_BUILD=0

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gguf)
      export TENSOR4J_GGUF_PATH="$2"
      shift 2
      ;;
    --template)
      export TENSOR4J_CHAT_TEMPLATE="$2"
      shift 2
      ;;
    --mode)
      export TENSOR4J_CHAT_MODE="$2"
      shift 2
      ;;
    --max-tokens)
      export TENSOR4J_CHAT_MAX_TOKENS="$2"
      shift 2
      ;;
    --save-dir)
      export TENSOR4J_CHAT_SAVE_DIR="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      ;;
  esac
done

# Sensible defaults for Llama 3.2 Instruct chat
export TENSOR4J_CHAT_TEMPLATE="${TENSOR4J_CHAT_TEMPLATE:-llama3}"
export TENSOR4J_CHAT_MODE="${TENSOR4J_CHAT_MODE:-quality}"
export TENSOR4J_CHAT_MAX_TOKENS="${TENSOR4J_CHAT_MAX_TOKENS:-256}"
export TENSOR4J_CHAT_SAVE_DIR="${TENSOR4J_CHAT_SAVE_DIR:-${HOME}/.local/conversations}"

if [[ -z "${TENSOR4J_GGUF_PATH:-}" ]]; then
  for candidate in \
    "${HOME}/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/"*"/Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
    "${HOME}/AppData/Local/Temp/Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
    "/tmp/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
  do
    # shellcheck disable=SC2086
    if [[ -f $candidate ]]; then
      export TENSOR4J_GGUF_PATH="$candidate"
      break
    fi
  done
fi

if [[ -z "${TENSOR4J_GGUF_PATH:-}" || ! -f "${TENSOR4J_GGUF_PATH}" ]]; then
  echo "error: set TENSOR4J_GGUF_PATH to your Llama 3.2 GGUF file, or pass --gguf PATH" >&2
  echo "  example: TENSOR4J_GGUF_PATH=~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf $0" >&2
  exit 1
fi

mkdir -p "${TENSOR4J_CHAT_SAVE_DIR}"

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  echo "Building tensor4j + chat-demo (skip with --skip-build)..." >&2
  (cd "${TENSOR4J_ROOT}" && mvn -q install -DskipTests)
  (cd "${SCRIPT_DIR}" && mvn -q package -DskipTests)
fi

# Heap for 1B Q4 model + KV cache (override with MAVEN_OPTS if needed)
export MAVEN_OPTS="${MAVEN_OPTS:-} -Xmx10g"

exec mvn -q exec:java -f "${SCRIPT_DIR}/pom.xml"
