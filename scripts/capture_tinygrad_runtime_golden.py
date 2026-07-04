#!/usr/bin/env python3
"""Capture tinygrad *runtime* sampling outputs into a checked-in JSON fixture.

Decoupled from Java/CI — run manually when tinygrad is installed, commit the JSON:

  pip install tinygrad
  python scripts/capture_tinygrad_runtime_golden.py

Writes: java/resources/tinygrad-runtime-sampling-golden.json

Uses real ``tinygrad.Tensor`` ops. ``extra/models/llama.py`` is vendored inline because
the PyPI wheel does not ship ``extra/``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "java/resources/tinygrad-runtime-sampling-golden.json"


def require_tinygrad():
    try:
        from tinygrad import Tensor  # noqa: F401
        import tinygrad  # noqa: F401
    except ImportError as exc:
        print("tinygrad not installed — pip install tinygrad", file=sys.stderr)
        raise SystemExit(2) from exc


def gumbel_apps_llm(logits: list[float], temperature: float, seed: int) -> int:
    """apps/llm.py: argmax(logits/temp - log(-log(uniform)))."""
    from tinygrad import Tensor

    Tensor.manual_seed(seed)
    t = Tensor(logits)
    temp = Tensor(max(temperature, 1e-12))
    u = Tensor.rand(t.shape)
    scores = t / temp - (u.maximum(1e-12).log().neg()).log()
    return int(scores.argmax().item())


def llama_sample_runtime(
    logits: list[float],
    temp: float,
    k: int,
    p: float,
    af: float,
    ap: float,
    alpha_counter,
    tensor_seed: int,
) -> int:
    """extra/models/llama.py sample() — 1d logits [vocab] in, scalar token out."""
    from tinygrad import Tensor, dtypes

    Tensor.manual_seed(tensor_seed)
    t = Tensor(logits)
    if temp < 1e-6:
        return int(t.argmax().item())

    if af or ap:
        if alpha_counter is None:
            alpha_counter = Tensor.zeros_like(t, dtype=dtypes.int32)
        logits_adj = t - (alpha_counter * af + (alpha_counter > 0) * ap)
    else:
        logits_adj = t

    logits_adj = (logits_adj != logits_adj).where(float("-inf"), logits_adj)
    probs = (logits_adj / temp).softmax()

    counter = Tensor.arange(probs.numel())
    counter2 = Tensor.arange(probs.numel() - 1, -1, -1)

    if k:
        output = Tensor.zeros(k)
        output_indices = Tensor.zeros(k, dtype=dtypes.int32)
        for i in range(k):
            t_max = probs.max()
            t_argmax = (probs.numel() - ((probs == t_max) * counter2).max() - 1).cast(dtypes.default_int)
            output = output + t_max.unsqueeze(0).pad(((i, k - i - 1),))
            output_indices = output_indices + t_argmax.unsqueeze(0).pad(((i, k - i - 1),))
            probs = (counter == t_argmax).where(0, probs)
        output_cumsum = output[::-1].cumsum()[::-1] + probs.sum()
        output = (output_cumsum >= (1 - p)) * output
        output_indices = (output_cumsum >= (1 - p)) * output_indices
        output_idx = output.multinomial()
        return int(output_indices[output_idx].item())

    return int(probs.multinomial().item())


def llama_sample(
    logits: list[float],
    temperature: float,
    top_k: int,
    top_p: float,
    alpha_f: float,
    alpha_p: float,
    alpha_counts: list[int] | None,
    tensor_seed: int,
) -> int:
    from tinygrad import Tensor

    counter = None
    if alpha_counts:
        counts = [0] * len(logits)
        for idx in alpha_counts:
            counts[idx] += 1
        counter = Tensor(counts)
    return llama_sample_runtime(
        logits, temperature, top_k, top_p, alpha_f, alpha_p, counter, tensor_seed
    )


def capture() -> dict[str, Any]:
    import tinygrad

    cases_out: list[dict[str, Any]] = []

    gumbel_specs = [
        ("gumbel_v3_seed42", [1.0, 2.0, 0.5], 0.7, 42),
        ("gumbel_v5_seed7", [1.0, 5.0, 4.0, 0.0, -10.0], 1.0, 7),
        ("gumbel_v4_low_temp", [1.0, 5.0, 2.0, 0.0], 0.01, 42),
    ]
    for name, logits, temp, seed in gumbel_specs:
        token = gumbel_apps_llm(logits, temp, seed)
        cases_out.append(
            {
                "name": name,
                "path": "apps_llm_gumbel",
                "logitsShape": [len(logits)],
                "logits": logits,
                "temperature": temp,
                "tensorSeed": seed,
                "runtimeToken": token,
            }
        )

    llama_specs = [
        ("llama_v4_tensor_seed_0", [0.0, 1.0, 2.0, 3.0], 1.0, 0, 0, 1.0, 0.0, 0.0, None),
        ("llama_v4_tensor_seed_99", [0.0, 1.0, 2.0, 3.0], 1.0, 99, 0, 1.0, 0.0, 0.0, None),
        ("llama_v5_topk2_topp09_seed_0", [1.0, 5.0, 4.0, 0.0, -10.0], 1.0, 0, 2, 0.9, 0.0, 0.0, None),
        ("llama_v5_topk2_topp09_seed_42", [1.0, 5.0, 4.0, 0.0, -10.0], 1.0, 42, 2, 0.9, 0.0, 0.0, None),
        ("llama_alpha_fp_seed_0", [0.0, 10.0, 10.0], 1.0, 0, 0, 1.0, 0.1, 0.5, [1, 1]),
        ("llama_alpha_fp_seed_17", [0.0, 10.0, 10.0], 1.0, 17, 0, 1.0, 0.1, 0.5, [1, 1]),
    ]
    for name, logits, temp, seed, top_k, top_p, af, ap, counts in llama_specs:
        token = llama_sample(logits, temp, top_k, top_p, af, ap, counts, seed)
        cases_out.append(
            {
                "name": name,
                "path": "llama_sample",
                "logitsShape": [len(logits)],
                "logits": logits,
                "temperature": temp,
                "tensorSeed": seed,
                "topK": top_k,
                "topP": top_p,
                "alphaFrequency": af,
                "alphaPresence": ap,
                "alphaCounts": counts,
                "runtimeToken": token,
            }
        )

    return {
        "captureVersion": 1,
        "tinygradVersion": getattr(tinygrad, "__version__", "unknown"),
        "cases": cases_out,
    }


def main() -> None:
    require_tinygrad()
    payload = capture()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['cases'])} runtime cases to {OUT}")
    for case in payload["cases"]:
        print(f"  {case['name']}: token {case['runtimeToken']} ({case['path']})")


if __name__ == "__main__":
    main()
