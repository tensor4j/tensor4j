#!/usr/bin/env python3
"""Generate tinygrad-sampling-golden.json using the same algorithms as Java reference classes.

Mirrors:
  - apps/llm.py Gumbel-max (Java Random seed for cross-language gumbel cases)
  - extra/models/llama.py sample() (multinomial roll cases)

Run from tensor4j/:  python scripts/gen_tinygrad_sampling_golden.py
Or Java (authoritative, uses java.util.Random for gumbel seeds):
  mvn test-compile && java -cp target/classes;target/test-classes com.github.tensor4j.models.chat.GenerateTinygradSamplingGoldenJson
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

MIN_UNIFORM = 1e-12


def standard_gumbel(u: float) -> float:
    u = max(u, MIN_UNIFORM)
    return -math.log(-math.log(u))


def gumbel_sample(logits: list[float], temp: float, seed: int) -> int:
    rng = random.Random(seed)
    t = max(temp, MIN_UNIFORM)
    best, best_score = 0, float("-inf")
    for i, logit in enumerate(logits):
        u = max(rng.random(), MIN_UNIFORM)
        score = logit / t + standard_gumbel(u)
        if score > best_score:
            best, best_score = i, score
    return best


def softmax(logits: list[float], temp: float) -> list[float]:
    t = max(temp, MIN_UNIFORM)
    m = max(logits)
    exps = [math.exp((x - m) / t) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def multinomial_at_roll(probs: list[float], roll: float) -> int:
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if roll <= cumulative:
            return i
    return max(range(len(probs)), key=lambda i: probs[i])


def llama_sample(
    logits: list[float],
    temp: float,
    top_k: int,
    top_p: float,
    alpha_f: float,
    alpha_p: float,
    alpha_counts: list[int],
    roll: float,
) -> int:
    if temp < 1e-6:
        return max(range(len(logits)), key=lambda i: logits[i])
    work = logits[:]
    for i in range(len(work)):
        c = alpha_counts[i] if i < len(alpha_counts) else 0
        if c > 0:
            work[i] -= c * alpha_f + alpha_p
    probs = softmax(work, temp)
    if top_k > 0 and top_k < len(probs):
        remaining = probs[:]
        top_probs, top_idx = [], []
        for _ in range(top_k):
            j = max(range(len(remaining)), key=lambda i: remaining[i])
            top_probs.append(remaining[j])
            top_idx.append(j)
            remaining[j] = 0.0
        tail = sum(remaining)
        kept_probs, kept_idx = [], []
        cumulative = tail
        for i in range(top_k - 1, -1, -1):
            cumulative += top_probs[i]
            if cumulative >= 1.0 - top_p:
                kept_probs.append(top_probs[i])
                kept_idx.append(top_idx[i])
        if not kept_probs:
            kept_probs, kept_idx = [top_probs[-1]], [top_idx[-1]]
        total = sum(kept_probs)
        target = roll * total
        cumulative = 0.0
        for i, p in enumerate(kept_probs):
            cumulative += p
            if target <= cumulative:
                return kept_idx[i]
        return kept_idx[-1]
    return multinomial_at_roll(probs, roll)


def main() -> None:
    cases = []

    gumbel_specs = [
        ("gumbel_v3_seed42", [1.0, 2.0, 0.5], 0.7, 42),
        ("gumbel_v5_seed7", [1.0, 5.0, 4.0, 0.0, -10.0], 1.0, 7),
        ("gumbel_v4_low_temp", [1.0, 5.0, 2.0, 0.0], 0.01, 42),
    ]
    for name, logits, temp, seed in gumbel_specs:
        token = gumbel_sample(logits, temp, seed)
        cases.append(
            {
                "name": name,
                "logits": logits,
                "temperature": temp,
                "seed": seed,
                "gumbelMax": True,
                "expectedToken": token,
            }
        )

    llama_specs = [
        ("llama_v4_roll_0_10", [0.0, 1.0, 2.0, 3.0], 1.0, 0, 1.0, 0.0, 0.0, [], 0.10),
        ("llama_v5_topk2_topp09_roll_0_99", [1.0, 5.0, 4.0, 0.0, -10.0], 1.0, 2, 0.9, 0.0, 0.0, [], 0.99),
        ("llama_alpha_fp_roll_0_50", [0.0, 10.0, 10.0], 1.0, 0, 1.0, 0.05, 0.1, [0, 2, 0], 0.50),
    ]
    for name, logits, temp, top_k, top_p, af, ap, counts, roll in llama_specs:
        token = llama_sample(logits, temp, top_k, top_p, af, ap, counts, roll)
        cases.append(
            {
                "name": name,
                "logits": logits,
                "temperature": temp,
                "topK": top_k,
                "topP": top_p,
                "alphaFrequency": af,
                "alphaPresence": ap,
                "alphaCounts": counts,
                "multinomialRoll": roll,
                "gumbelMax": False,
                "expectedToken": token,
            }
        )

    out = Path(__file__).resolve().parents[1] / "java/test/resources/tinygrad-sampling-golden.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": cases}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
