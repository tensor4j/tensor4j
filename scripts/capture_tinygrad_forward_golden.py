#!/usr/bin/env python3
"""Capture forward logits golden JSON for tensor4j ChatModel fixtures.

Regenerate when forward math changes:

  python scripts/capture_tinygrad_forward_golden.py

Writes: java/resources/tinygrad-forward-golden.json

Note: primary CI goldens are computed in Java ({@code TinygradForwardGoldenCases})
because chat forward runs on the llama runtime track. This script documents the
intended capture workflow for cross-checking against tinygrad when available.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "java/resources/tinygrad-forward-golden.json"


def capture() -> dict:
    return {
        "captureVersion": 1,
        "note": "Populate from Java TinygradForwardGoldenCases or tinygrad Transformer.forward when wired",
        "cases": [],
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(capture(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote placeholder to {OUT}")


if __name__ == "__main__":
    main()
