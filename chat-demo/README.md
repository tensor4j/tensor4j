# chat-demo

Standalone Maven project demonstrating the **ggml / GGUF chat track**: load weights,
tokenize a prompt, run llama-style forward, and greedy-generate a short completion.

Runs as **Failsafe integration tests** (`*IT.java`) via `mvn verify`.

## Prerequisites

Install tensor4j into the local Maven repository:

```bash
cd ..
mvn install -DskipTests
```

## Run (automated demo)

```bash
cd chat-demo
mvn verify
```

Prints a transcript: model shape, token ids, logits check, sampled completions.

**Default sampling is quality mode** (temperature 0.7, top-p 0.9, top-k 40, alpha frequency/presence penalties, Gumbel-max).

| Variable | Values | Default |
|----------|--------|---------|
| `TENSOR4J_CHAT_MODE` | `quality`, `greedy` | `quality` |
| `TENSOR4J_CHAT_RNG` | `secure`, `legacy` | `secure` |
| `TENSOR4J_CHAT_TEMPERATURE` | float | `0.7` |
| `TENSOR4J_CHAT_TOP_P` | float | `0.9` |
| `TENSOR4J_CHAT_TOP_K` | int | `40` |
| `TENSOR4J_CHAT_MIN_TOKENS` | int | `2` |
| `TENSOR4J_CHAT_MAX_TOKENS` | int | `128` |
| `TENSOR4J_CHAT_SEED` | long | `42` (legacy RNG only) |
| `TENSOR4J_CHAT_ALPHA_F` | float | `0.05` |
| `TENSOR4J_CHAT_ALPHA_P` | float | `0.1` |
| `TENSOR4J_CHAT_GUMBEL` | `true`, `false` | `true` |
| `TENSOR4J_CHAT_PREFILL_CHUNK` | int | `128` |
| `TENSOR4J_CHAT_TEMPLATE` | `plain`, `llama3` | `plain` |

Use `TENSOR4J_CHAT_MODE=greedy` for fast argmax smoke only. Set `TENSOR4J_CHAT_GUMBEL=false` to use top-p nucleus sampling instead of Gumbel-max.

Set `TENSOR4J_CHAT_RNG=legacy` for seeded `java.util.Random` draws (reproducible completions via `TENSOR4J_CHAT_SEED`). Default is `SecureRandom`.

**Level 12 chat demo fixture** (`buildChatDemoModel()`): 768-dim embeddings, 12 layers, 12 heads (6 KV), 2048 context, 192-token tinygrad-style llama3 vocab chain, up to 128 new tokens per reply.

| Scale | Smoke fixtures | Chat demo (level 12) |
|-------|----------------|----------------------|
| embd | 32 | 768 |
| layers | 1 | 12 |
| ctx | 8 | 2048 |
| vocab | 4–5 | 192 |
| max new tokens | 32 | 128 |

**Sampling RNG** (tinygrad `apps/llm.py`): Gumbel-max uses uniform draws + `-log(-log(u))`; multinomial uses unit-interval rolls. Default entropy is `SecureRandom` via `ChatSamplingRngMode.SECURE`. Use `LEGACY` + `TENSOR4J_CHAT_SEED` for deterministic tests. See `ChatSamplingRng`, `ChatDemoVocab`, and unit tests.

**Four-turn demo**: `FourTurnChatDemoIT` runs 4 user/bot exchanges with KV prefix reuse.

## Optional: real GGUF weights

```bash
export TENSOR4J_GGUF_PATH=/path/to/model.gguf   # Windows: set TENSOR4J_GGUF_PATH=...
mvn verify
```

Runs `ExternalGgufChatDemoIT` in addition to the mini fixture demo.

## Interactive chat (30 minute session)

For a manual REPL (stdin), enable the interactive integration test:

```bash
export TENSOR4J_CHAT_INTERACTIVE=1
# optional: export TENSOR4J_GGUF_PATH=/path/to/model.gguf
mvn verify -DskipTests
```

- Reads lines from stdin (`you>` / `bot>`)
- Type `exit` to quit early
- JUnit `@Timeout` ends the session after **30 minutes**
- Skipped in normal CI (env var not set)

## Scenarios

| Class | When | Story |
|-------|------|-------|
| `MiniChatDemoIT` | always | Level-12 fixture: quality sampling (seeded legacy in CI) |
| `FourTurnChatDemoIT` | always | 4-turn Hello session + KV prefix reuse |
| `ExternalGgufChatDemoIT` | `TENSOR4J_GGUF_PATH` set | mmap real weights, short generation |
| `InteractiveChatDemoIT` | `TENSOR4J_CHAT_INTERACTIVE=1` | blocking REPL, 30 min max |

## Layout

```
chat-demo/
  pom.xml
  src/test/java/          *IT.java
  src/test/resources/
    demo-prompts.txt
```
