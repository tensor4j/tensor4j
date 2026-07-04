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

**Default sampling is quality mode** (temperature 0.7, top-p 0.9, suppress early EOS).

| Variable | Values | Default |
|----------|--------|---------|
| `TENSOR4J_CHAT_MODE` | `quality`, `greedy` | `quality` |

Use `TENSOR4J_CHAT_MODE=greedy` for fast argmax smoke only.

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
| `MiniChatDemoIT` | always | Mini chat-tuned GGUF: quality sampling (default) |
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
