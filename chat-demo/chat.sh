#!/usr/bin/env bash
# Interactive chat via tensor4j.
# Default: Qwen2.5-1.5B Instruct Q4_K_M (~1 GB, bigger than Llama 3.2 1B).
# Use --llama for Llama 3.2 1B Instruct instead.
#
# Saves each session to ~/.local/conversations/<timestamp>-<model>-chat.md (readable)
# and <timestamp>-<model>-chat-audit.log (token audit) — updated after every turn.
#
# Usage:
#   ./chat.sh                        # greedy / 32 tokens — same defaults as tensor4j-gguf-it
#   ./chat-enhanced.sh               # quality / 256 tokens / focus — open-ended, may ask questions
#   ./chat.sh --download              # fetch default Qwen GGUF if missing
#   ./chat.sh --llama                 # Llama 3.2 1B instead of Qwen
#   ./chat.sh --gguf /path/to/model.gguf
#   TENSOR4J_CHAT_MAX_TOKENS=256 ./chat.sh
#
# Options (also available as env vars):
#   --llama              use Llama 3.2 1B (template llama3, default log base llama32)
#   --download           download default Qwen GGUF from Hugging Face if missing
#   --legacy             tensor4j/tinygrad history (full token replay); default is llama.cpp delta mode
#   --gguf PATH          TENSOR4J_GGUF_PATH
#   --template NAME      TENSOR4J_CHAT_TEMPLATE (qwen2 default, llama3 with --llama)
#   --mode MODE          TENSOR4J_CHAT_MODE (quality|greedy)
#   --max-tokens N       TENSOR4J_CHAT_MAX_TOKENS
#   --save-dir PATH      TENSOR4J_CHAT_SAVE_DIR (default ~/.local/conversations)
#   --build              run mvn install before chat (default: skip build)
#   --skip-build         same as default; kept for compatibility
#   -h, --help           show help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSOR4J_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SKIP_BUILD=1
DOWNLOAD=0
USE_LLAMA=0

usage() {
  sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llama)
      USE_LLAMA=1
      shift
      ;;
    --download)
      DOWNLOAD=1
      shift
      ;;
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
    --legacy)
      export TENSOR4J_CHAT_HISTORY_MODE=legacy
      shift
      ;;
    --build)
      SKIP_BUILD=0
      shift
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

export TENSOR4J_CHAT_MODE="${TENSOR4J_CHAT_MODE:-greedy}"
export TENSOR4J_CHAT_MAX_TOKENS="${TENSOR4J_CHAT_MAX_TOKENS:-512}"
export TENSOR4J_CHAT_MIN_TOKENS="${TENSOR4J_CHAT_MIN_TOKENS:-0}"
export TENSOR4J_CHAT_SAVE_DIR="${TENSOR4J_CHAT_SAVE_DIR:-${HOME}/.local/conversations}"
export TENSOR4J_CHAT_HISTORY_MODE="${TENSOR4J_CHAT_HISTORY_MODE:-llama}"
export TENSOR4J_CHAT_DEFAULT_SYSTEM="${TENSOR4J_CHAT_DEFAULT_SYSTEM:-true}"
export TENSOR4J_CHAT_SYSTEM_PROMPT="${TENSOR4J_CHAT_SYSTEM_PROMPT:-focus}"
export TENSOR4J_CHAT_KV_CACHE="${TENSOR4J_CHAT_KV_CACHE:-false}"
export TENSOR4J_CHAT_DEBUG="${TENSOR4J_CHAT_DEBUG:-false}"
MODELS_DIR="${HOME}/.local/models"

find_llama_gguf() {
  for candidate in \
    "${TENSOR4J_GGUF_PATH:-}" \
    "${HOME}/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/"*"/Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
    "${HOME}/AppData/Local/Temp/Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
    "/tmp/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
  do
    # shellcheck disable=SC2086
    if [[ -n "${candidate}" && -f $candidate ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

find_qwen_gguf() {
  local qwen_file="${TENSOR4J_QWEN_GGUF_FILE:-qwen2.5-1.5b-instruct-q4_k_m.gguf}"
  for candidate in \
    "${TENSOR4J_GGUF_PATH:-}" \
    "${TENSOR4J_QWEN_GGUF:-}" \
    "${MODELS_DIR}/${qwen_file}" \
    "${MODELS_DIR}/qwen2.5-3b-instruct-q4_k_m.gguf" \
    "${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct-GGUF/snapshots/"*"/${qwen_file}" \
    "${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct-GGUF/snapshots/"*"/qwen2.5-3b-instruct-q4_k_m.gguf"
  do
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

download_qwen_gguf() {
  if ! command -v huggingface-cli >/dev/null 2>&1 && ! command -v hf >/dev/null 2>&1; then
    echo "error: Qwen GGUF not found. Install huggingface_hub and run: $0 --download" >&2
    echo "  pip install huggingface_hub" >&2
    exit 1
  fi
  local repo="${TENSOR4J_QWEN_REPO:-Qwen/Qwen2.5-1.5B-Instruct-GGUF}"
  local file="${TENSOR4J_QWEN_GGUF_FILE:-qwen2.5-1.5b-instruct-q4_k_m.gguf}"
  local hf_cli
  hf_cli="$(command -v hf || command -v huggingface-cli)"
  mkdir -p "${MODELS_DIR}"
  echo "Downloading ${repo} ${file} (~1 GB)..." >&2
  PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}" \
    "${hf_cli}" download "${repo}" "${file}" --local-dir "${MODELS_DIR}"
  export TENSOR4J_GGUF_PATH="${MODELS_DIR}/${file}"
}

if [[ "${USE_LLAMA}" -eq 1 ]]; then
  export TENSOR4J_CHAT_TEMPLATE="${TENSOR4J_CHAT_TEMPLATE:-llama3}"
  export TENSOR4J_CHAT_LOG_BASE="${TENSOR4J_CHAT_LOG_BASE:-llama32}"
  if [[ -z "${TENSOR4J_GGUF_PATH:-}" ]]; then
    TENSOR4J_GGUF_PATH="$(find_llama_gguf || true)"
    export TENSOR4J_GGUF_PATH
  fi
  if [[ -z "${TENSOR4J_GGUF_PATH:-}" || ! -f "${TENSOR4J_GGUF_PATH}" ]]; then
    echo "error: Llama GGUF not found. Set TENSOR4J_GGUF_PATH or pass --gguf PATH" >&2
    echo "  example: $0 --llama --gguf ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf" >&2
    exit 1
  fi
  echo "Using Llama GGUF: ${TENSOR4J_GGUF_PATH}" >&2
else
  export TENSOR4J_CHAT_TEMPLATE="${TENSOR4J_CHAT_TEMPLATE:-qwen2}"
  export TENSOR4J_CHAT_LOG_BASE="${TENSOR4J_CHAT_LOG_BASE:-qwen25-1.5b}"
  if [[ "${DOWNLOAD}" -eq 1 ]]; then
    download_qwen_gguf
  elif [[ -z "${TENSOR4J_GGUF_PATH:-}" ]]; then
    export TENSOR4J_GGUF_PATH="$(find_qwen_gguf || true)"
  fi
  if [[ -z "${TENSOR4J_GGUF_PATH:-}" || ! -f "${TENSOR4J_GGUF_PATH}" ]]; then
    echo "error: Qwen GGUF not found. Run: $0 --download" >&2
    exit 1
  fi
  echo "Using Qwen GGUF: ${TENSOR4J_GGUF_PATH}" >&2
fi

mkdir -p "${TENSOR4J_CHAT_SAVE_DIR}"

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  echo "Building tensor4j + chat-demo..." >&2
  (cd "${TENSOR4J_ROOT}" && mvn -q install -DskipTests)
  (cd "${SCRIPT_DIR}" && mvn -q package -DskipTests)
fi

export MAVEN_OPTS="${MAVEN_OPTS:-} -Xmx10g"

exec mvn -q exec:java -f "${SCRIPT_DIR}/pom.xml"
