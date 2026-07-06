#!/usr/bin/env python3
"""Compare tensor4j ChatML token ids against llama.cpp llama_chat_apply_template."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLAMA_BIN = ROOT / "vendor" / "llama.cpp" / "build" / "bin" / "Release" / "llama-tensor4j-chat-tokens.exe"
LLAMA_BIN_ALT = ROOT / "vendor" / "llama.cpp" / "build" / "bin" / "llama-tensor4j-chat-tokens"


def parse_ids(label: str, text: str) -> list[int] | None:
    match = re.search(rf"{re.escape(label)}=\[([^\]]*)\]", text)
    if not match:
        return None
    raw = match.group(1).strip()
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",")]


def run_llama(gguf: Path) -> str:
    exe = LLAMA_BIN if LLAMA_BIN.is_file() else LLAMA_BIN_ALT
    if not exe.is_file():
        raise SystemExit(f"Build llama probe first: {exe}")
    proc = subprocess.run(
        [str(exe), "-m", str(gguf), "-s", "all"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def run_tensor4j(gguf: Path) -> str:
    env = os.environ.copy()
    env["TENSOR4J_GGUF_PATH"] = str(gguf)
    proc = subprocess.run(
        [
            "mvn",
            "-q",
            "-DincludeScope=test",
            "-Dexec.classpathScope=test",
            "org.codehaus.mojo:exec-maven-plugin:3.1.0:java",
            "-Dexec.mainClass=com.github.tensor4j.models.chat.reference.LiveQwenChatTokenProbe",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.stdout


def compare(label: str, llama: dict[str, list[int]], tensor4j: dict[str, list[int]]) -> bool:
    left = llama.get(label)
    right = tensor4j.get(label)
    if left is None or right is None:
        print(f"MISSING {label}: llama={left is not None} tensor4j={right is not None}")
        return False
    if left == right:
        print(f"OK {label} ({len(left)} tokens)")
        return True
    print(f"MISMATCH {label}: llama={len(left)} tensor4j={len(right)}")
    n = min(len(left), len(right))
    for i in range(n):
        if left[i] != right[i]:
            print(f"  first diff at {i}: llama={left[i]} tensor4j={right[i]}")
            break
    if len(left) != len(right):
        print(f"  llama tail:   {left[n:min(n + 12, len(left))]}")
        print(f"  tensor4j tail:{right[n:min(n + 12, len(right))]}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gguf",
        default=os.environ.get(
            "TENSOR4J_GGUF_PATH",
            str(Path.home() / ".local" / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        ),
    )
    args = parser.parse_args()
    gguf = Path(args.gguf)
    if not gguf.is_file():
        print(f"GGUF not found: {gguf}", file=sys.stderr)
        return 1

    llama_out = run_llama(gguf)
    java_out = run_tensor4j(gguf)
    print("=== llama.cpp ===")
    print(llama_out, end="")
    print("=== tensor4j ===")
    print(java_out, end="")

    llama = {
        "turn1_prefill": parse_ids("llama_turn1_prefill", llama_out),
        "turn1_closed": parse_ids("llama_turn1_closed", llama_out),
        "turn2_delta": parse_ids("llama_turn2_delta", llama_out),
        "turn2_full": parse_ids("llama_turn2_full", llama_out),
    }
    tensor4j = {
        "turn1_prefill": parse_ids("tensor4j_turn1_prefill", java_out),
        "turn1_closed": parse_ids("tensor4j_turn1_closed", java_out),
        "turn2_delta": parse_ids("tensor4j_turn2_delta", java_out),
        "turn2_full": parse_ids("tensor4j_turn2_full", java_out),
        "turn2_char_delta": parse_ids("tensor4j_turn2_char_delta", java_out),
    }

    ok = True
    ok &= compare("turn1_prefill", llama, tensor4j)
    ok &= compare("turn1_closed", llama, tensor4j)
    ok &= compare("turn2_delta", llama, tensor4j)
    ok &= compare("turn2_full", llama, tensor4j)
    char_delta = tensor4j.get("turn2_char_delta")
    token_delta = tensor4j.get("turn2_delta")
    if char_delta is not None and token_delta is not None:
        if char_delta == token_delta:
            print("OK tensor4j char_delta == token_delta")
        else:
            print("MISMATCH tensor4j char_delta vs token_delta (llama uses char prev_len)")
            ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
