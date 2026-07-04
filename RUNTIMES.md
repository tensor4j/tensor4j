# tensor4j runtimes



tensor4j hosts **two independent math/runtime tracks**. They share the repo and the algebra regression suite, but **must not share implementation code**.



## Rule: do not cross the streams



| Track | Vendor study copy | Java packages | Weight format | Primary use |

|-------|-------------------|---------------|---------------|-------------|

| **tinygrad** | `vendor/tinygrad/` | `core/`, `autograd/`, `core/lazy/`, `nn/`, `io/` (`.t4j.json`) | `.t4j.json` | Training, autograd, CNN/MLP ops, algebra model |

| **ggml/GGUF** | `vendor/llama.cpp/` | `runtime/ggml/`, `runtime/gguf/` | `.gguf` | CPU/RAM inference, chat-scale shapes |



- **Never** alter tinygrad-track math to accommodate ggml/GGUF needs.

- **Never** import `runtime.*` from `core`, `autograd`, or `lazy`.

- New ggml work goes under `runtime/ggml/`, `runtime/gguf/` (and future `models/chat/`).

- Vendor trees are **local study copies** (gitignored, not shipped in the JAR).



## tinygrad track (current)



Reference: [tinygrad](https://github.com/tinygrad/tinygrad) → `vendor/tinygrad/`



- Float32 eager `Tensor` + lazy `LazyUOp` DAG, autograd, parity fixtures

- Algebra MLP (`models/algebra/`) is the **canonical end-to-end test case**

- Export/import via tinygrad-compatible `.t4j.json`



Clone (optional):



```bash

git clone --depth 1 https://github.com/tinygrad/tinygrad.git vendor/tinygrad

```



## ggml / GGUF track (in progress)



Reference: [llama.cpp](https://github.com/ggml-org/llama.cpp) → `vendor/llama.cpp/`



| Area | llama.cpp paths | Java package | Status |

|------|-----------------|--------------|--------|

| **Tensor layout** | `ggml/include/ggml.h`, `ggml/src/gguf.cpp` | `runtime/ggml/` | `GgmlType`, `GgmlTensorShape`, `GgmlLayout`, `GgmlQuant` |

| **GGUF metadata** | `ggml/include/gguf.h`, `ggml/src/gguf.cpp` | `runtime/gguf/` | `GgufReader`, `GgufWriter`, `GgufFile`, weight layout |

| **Graph / ops** | `src/llama-graph.cpp`, `ggml-cpu/ops.cpp` | `runtime/infer/`, `runtime/graph/` | `GgmlOps`, `Rope`, `InferGraph`, `LlamaAttentionForward`, `LlamaBlockForward` |
| **Unicode / BPE split** | `src/unicode.cpp`, `src/llama-vocab.cpp` | `runtime/unicode/` | `UnicodeRegexSplit`, collapsed-unicode + custom splitters (**default:** `BpeSplitMode.LLAMA_UNICODE`; opt-in `java_regex` via `-Dtensor4j.bpe.split=java_regex`) |

| **Memory** | `src/llama-kv-cache.h` | `runtime/memory/` | `KvCache`, `RingKvCache` |

| **Chat models** | `src/llama-model.cpp`, `src/llama-vocab.cpp` | `models/chat/` | `ChatModel`, `ChatTokenizer`, `BpePreType` (llama3/qwen2/gemma4/…), lazy `InferWeight` |



Clone (optional):



```bash

git clone --depth 1 https://github.com/ggml-org/llama.cpp.git vendor/llama.cpp

```



## Test strategy



1. **Algebra model** (`models/algebra/`, `.t4j.json`, 233+ unit tests) — always green; tinygrad track only.

2. **ggml track** — tests under `runtime/ggml/` and `runtime/gguf/`; no imports from `core` math.

3. Chat inference integration tests use synthetic GGUF fixtures; **`ChatGgufIntegrationTest`** runs encode → forward → sample in CI; optional real-model smoke via `TENSOR4J_GGUF_PATH` (`ExternalGgufSmokeTest`).
4. **Golden tokenizer vectors** — `TokenizerGoldenFixtures` + `UnicodeGoldenSplitTest` lock llama.cpp pre-split parity for llama3/qwen2/kimi-k2/afmoe.



## Coding style (both tracks)



Same project conventions: no lambdas in sample code, plain loops, JUnit 5, `TensorAssert.assertAllClose` for floats.


