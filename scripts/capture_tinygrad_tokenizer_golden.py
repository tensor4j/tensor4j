#!/usr/bin/env python3
"""Capture tinygrad SimpleTokenizer encode/decode goldens into JSON.

Run manually when tinygrad is installed:

  pip install tinygrad
  python scripts/capture_tinygrad_tokenizer_golden.py

Writes: java/resources/tinygrad-tokenizer-golden.json

Vendored inline from apps/llm.py SimpleTokenizer (PyPI wheel lacks extra/).
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "java/resources/tinygrad-tokenizer-golden.json"


def ucat_range(pre: str) -> str:
    return "".join(re.escape(chr(cp)) for cp in range(0x323B0) if unicodedata.category(chr(cp)).startswith(pre))


class SimpleTokenizer:
    def __init__(self, normal_tokens: dict[str, int], special_tokens: dict[str, int], preset: str = "llama3"):
        bs = [*range(33, 127), *range(161, 173), *range(174, 256)]
        self._byte_decoder = {chr(b): b for b in bs} | {chr(256 + i): b for i, b in enumerate(b for b in range(256) if b not in bs)}
        r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
        self._split_to_word = re.compile(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
            + f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+"
        )
        self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")
        self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
        self._special_tokens = special_tokens
        self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
        self.preset = preset

    def _encode_word(self, word: bytes) -> list[int]:
        if (early := self._normal_tokens.get(word)) is not None:
            return [early]
        parts = [bytes([b]) for b in word]
        while True:
            i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j] + parts[j + 1], sys.maxsize), j) for j in range(len(parts) - 1)])[1]
            if i == -1:
                break
            parts[i : i + 2] = [parts[i] + parts[i + 1]]
        return [self._normal_tokens[p] for p in parts]

    def _encode_sentence(self, chunk: str) -> list[int]:
        return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]

    def encode(self, text: str) -> list[int]:
        tokens: list[int] = []
        pos = 0
        for match in self._split_to_sentence.finditer(text):
            tokens.extend(self._encode_sentence(text[pos : match.start()]) + [self._special_tokens[text[match.start() : match.end()]]])
            pos = match.end()
        return tokens + self._encode_sentence(text[pos:])

    def decode(self, ids: list[int]) -> str:
        return b"".join(self._tok2bytes[tid] for tid in ids).decode(errors="replace")

    def role(self, role: str) -> list[int]:
        if self.preset == "olmo":
            return self.encode("<|" + role + "|>\n")
        if self.preset == "qwen2":
            return self.encode("<|im_start|>" + role + "\n")
        return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")

    def end_turn(self, eos_id: int) -> list[int]:
        if self.preset == "olmo":
            return self.encode("\n")
        if self.preset == "qwen2":
            return [eos_id] + self.encode("\n")
        return [eos_id]


def qwen2_fixture_vocab() -> tuple[dict[str, int], dict[str, int], str]:
    im_end = "<|" + "im_end" + "|>"
    tokens = [
        "<|endoftext|>",
        "Hello",
        "<|im_start|>",
        "user",
        "assistant",
        "system",
        "\n",
        im_end,
    ]
    normal: dict[str, int] = {}
    special: dict[str, int] = {}
    for i, tok in enumerate(tokens):
        if tok.startswith("<|") and tok.endswith("|>") or tok == "\n":
            special[tok] = i
        else:
            normal[tok] = i
    return normal, special, "qwen2"


def fixture_vocab() -> tuple[dict[str, int], dict[str, int], str]:
    tokens = ["<s>", "Hello", "a", "b", "</s>"]
    normal = {t: i for i, t in enumerate(tokens)}
    return normal, {}, "llama3"


def capture() -> dict:
    normal, special, preset = fixture_vocab()
    tok = SimpleTokenizer(normal, special, preset)
    q_normal, q_special, q_preset = qwen2_fixture_vocab()
    qtok = SimpleTokenizer(q_normal, q_special, q_preset)
    im_end_id = q_special["<|" + "im_end" + "|>"]
    cases = [
        {"name": "encode_hello", "kind": "encode", "text": "Hello", "ids": tok.encode("Hello")},
        {"name": "decode_ab", "kind": "decode", "ids": [1, 2], "text": tok.decode([1, 2])},
        {"name": "role_user", "kind": "role", "role": "user", "ids": tok.role("user")},
        {"name": "qwen2_role_user", "kind": "role", "role": "user", "ids": qtok.role("user"), "preset": "qwen2"},
        {"name": "qwen2_role_system", "kind": "role", "role": "system", "ids": qtok.role("system"), "preset": "qwen2"},
        {"name": "qwen2_role_assistant", "kind": "role", "role": "assistant", "ids": qtok.role("assistant"), "preset": "qwen2"},
        {"name": "qwen2_end_turn", "kind": "end_turn", "ids": qtok.end_turn(im_end_id), "preset": "qwen2"},
        {
            "name": "qwen2_system_turn_default",
            "kind": "system_turn",
            "text": "You are a helpful assistant.",
            "ids": qtok.role("system") + qtok.end_turn(im_end_id),
            "preset": "qwen2",
        },
        {
            "name": "qwen2_prompt_turn1_hello",
            "kind": "prompt_turn1",
            "text": "Hello",
            "ids": qtok.role("system")
            + qtok.end_turn(im_end_id)
            + qtok.role("user")
            + qtok.encode("Hello")
            + qtok.end_turn(im_end_id)
            + qtok.role("assistant"),
            "preset": "qwen2",
        },
        {
            "name": "qwen2_user_turn_hello",
            "kind": "user_turn",
            "text": "Hello",
            "role": "user",
            "ids": qtok.role("user") + qtok.encode("Hello") + qtok.end_turn(im_end_id),
            "preset": "qwen2",
        },
    ]
    return {"captureVersion": 1, "preset": preset, "cases": cases}


def main() -> None:
    payload = capture()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['cases'])} cases to {OUT}")


if __name__ == "__main__":
    main()
