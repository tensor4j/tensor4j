#!/usr/bin/env bash
# Convert CRLF line endings to LF in all Java and JSON files under the repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v dos2unix >/dev/null 2>&1; then
  echo "error: dos2unix not found; install it (e.g. apt install dos2unix)" >&2
  exit 1
fi

mapfile -d '' files < <(find "$ROOT" -type f \( -name '*.java' -o -name '*.json' \
-o -name 'LICENsE' -o -name 'MANIFEST.MF' \) -print0)

if ((${#files[@]} == 0)); then
  echo "No .java or .json files found under $ROOT"
  exit 0
fi

dos2unix "${files[@]}"
echo "Converted ${#files[@]} Java/JSON file(s) under $ROOT"
