# tensor4j

Float32 tensor engine for Tensor4j: **tensor · autograd · lazy DAG · inference**.

Apache-2.0 (see `LICENSE`). Two **independent runtime tracks** — see [RUNTIMES.md](RUNTIMES.md):

- **tinygrad track** — `core/`, `autograd/`, lazy UOp DAG, training, `.safetensors` / `.t4j.json` weights, algebra model
- **ggml track** — `runtime/ggml/`, `runtime/gguf/`, GGUF, CPU/RAM inference study (vendor: `vendor/llama.cpp/`)

Vendor folders are optional local clones (gitignored, not redistributed in the JAR).

## Testing

See [TESTING.md](TESTING.md). Quick run from `sample-apps`:

```bash
npm run test:tensor4j
```

Algebra end-to-end demo — standalone Maven project:

```bash
cd tensor4j
mvn install -DskipTests
cd algebra-demo
mvn verify
```

See [algebra-demo/README.md](algebra-demo/README.md).

## Coding style

tensor4j follows **Tinygrad-style composition without lambdas**:

- Tensor ops chain as methods: `x.matmul(w).add(b).relu()`
- Autograd is internal; no function-passing in user code
- UI and CLI use named `ActionListener` / `Runnable` classes (no Java lambdas)
- No streams in sample code — plain loops for clarity and debuggability

## Layout

```
tensor4j/
  java/src/com/github/tensor4j/
    core/          Tensor, Shape (tinygrad track — do not mix llama math)
    autograd/      Function, GradFlow (tinygrad track)
    core/lazy/     LazyUOp DAG (tinygrad track)
    runtime/ggml/   ggml tensor types, shapes, byte layout, quant
    runtime/gguf/   GGUF metadata reader/writer, GgufWeightLoader, MmappedGgufFile
    runtime/infer/  F32 inference kernels (mul_mat, rms_norm, silu, RoPE)
    runtime/graph/  forward graph + llama attention/block
    runtime/memory/ KV cache (growable + ring eviction)
    models/chat/    ChatModel forward (embed → blocks → lm_head)
    nn/            Module, Linear, Sequential
    manifold/      EuclideanManifold, ManifoldPoint
    io/            ModelLoader (.t4j.json, .safetensors — tinygrad state-dict keys)
    models/algebra/   canonical regression test (ax + b = c MLP)
    cli/           CliRunner
    ui/            Swing desktop
  java/resources/models/   bundled weights
  algebra-demo/          Failsafe integration demos (mvn verify)
  tools/           tinygrad export script + Java weight generator
  vendor/tinygrad/   optional local clone (gitignored)
  vendor/llama.cpp/  optional local clone — GGUF, ggml, CPU/RAM study (gitignored)
  RUNTIMES.md      dual-runtime boundaries and study map
```

## CLI

```
tensor4j info
tensor4j infer --equation "2x + 3 = 11"
tensor4j train --epochs 60 --lr 0.05
tensor4j export --out algebra-v1.safetensors
tensor4j export --out algebra-v1.t4j.json [--format t4j_json]
tensor4j tensor --shape 2,2
```

## Vendor study clones

```bash
cd tensor4j
git clone --depth 1 https://github.com/tinygrad/tinygrad.git vendor/tinygrad
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git vendor/llama.cpp
```

## tinygrad training → Java inference (algebra test case)

```bash
cd tensor4j
python tools/export_algebra_model.py java/resources/models/algebra-v1.safetensors
python tools/export_algebra_model.py java/resources/models/algebra-v1.t4j.json --format t4j_json
cd ..
npm run build:jnlp:tensor4j
```

## JNLP

`http://127.0.0.1:4200/jnlp/tensor4j/app.jnlp` — GUI by default; pass CLI args via `javawsc` if needed.
