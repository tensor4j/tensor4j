#!/usr/bin/env python3
"""Capture tinygrad Q4_0 dequant golden vectors (optional runtime parity).

Requires: pip install tinygrad numpy

Run from tensor4j/:
  python scripts/capture_tinygrad_q4_dequant_golden.py

Writes java/resources/tinygrad-q4-dequant-golden.json when tinygrad is available.
Java tests use inline golden cases in TinygradQ4DequantGoldenCases (no Python in CI).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "java" / "resources" / "tinygrad-q4-dequant-golden.json"


def main() -> None:
    try:
        import numpy as np
        from tinygrad import Tensor
        from tinygrad.nn.state import ggml_data_to_tensor
    except ImportError as exc:
        print("skip: tinygrad not installed:", exc)
        return

    cases = []

    # Hardcoded block: scale=2.0, nibbles 8..23
    block = bytearray(18)
    block[0:2] = np.float16(2.0).tobytes()
    for j in range(16):
        low = 8 + j
        high = 8 + j + 16
        block[2 + j] = low | (high << 4)
    expected = [(i % 16) * 2.0 for i in range(32)]
    t = Tensor(bytes(block))
    dq = ggml_data_to_tensor(t, 32, 2).numpy().flatten().tolist()
    cases.append({
        "name": "single_block_scale_2",
        "shape": [32],
        "quant_bytes_hex": bytes(block).hex(),
        "expected": dq,
        "tolerance": 1e-5,
    })
    np.testing.assert_allclose(dq, expected, rtol=0, atol=1e-4)

    # Round-trip matrix [32, 2]
    source = [float(np.sin(i * 0.31)) for i in range(64)]
    from test.unit.test_gguf import quantize, dequantize, GGMLQuantizationType

    q_data = quantize(np.array(source, dtype=np.float32), GGMLQuantizationType.Q4_0)
    ref = dequantize(q_data, GGMLQuantizationType.Q4_0)
    q_tensor = Tensor(q_data.tobytes())
    dq = ggml_data_to_tensor(q_tensor, 64, 2).numpy().flatten().tolist()
    np.testing.assert_equal(dq, ref)
    cases.append({
        "name": "matrix_32x2_roundtrip",
        "shape": [32, 2],
        "quant_bytes_hex": q_data.tobytes().hex(),
        "expected": ref.tolist(),
        "tolerance": 0.2,
    })

    OUT.write_text(json.dumps({"format": "tensor4j-q4-dequant-v1", "cases": cases}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
