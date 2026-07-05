#!/usr/bin/env bash
# Back-compat wrapper — default chat is now Qwen via chat.sh.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/chat.sh" "$@"
