# algebra-demo

Standalone Maven project demonstrating the tensor4j **algebra MLP** on the tinygrad track:
load safetensors weights, infer `x` from `ax + b = c`, and run a train → export → reload loop.

Each scenario prints **worked algebra** (symbolic steps) alongside the **MLP prediction** so you
can compare classical solution with neural inference.
Uses standard Maven layout and **Failsafe** (`*IT.java`) — run with `mvn verify`, not raw `java -cp`.

## Prerequisites

Install the parent library into the local Maven repository (once, or after tensor4j changes):

```bash
cd ..
mvn install -DskipTests
```

## Run

```bash
cd algebra-demo
mvn verify
```

- **Surefire** (`test` phase): skipped — this module has integration tests only
- **Failsafe** (`integration-test` + `verify`): runs `*IT.java` and prints a demo transcript

Integration tests only (skip unit-test phase):

```bash
mvn verify -DskipTests
```

## Layout

```
algebra-demo/
  pom.xml
  src/test/java/          *IT.java demo scenarios
  src/test/resources/
    demo-equations.txt    equation fixture
    models/
      algebra-v1.safetensors   bundled weights (copy of ../java/resources/models/)
```

## Scenarios

| Class | Story |
|-------|-------|
| `BundledInferenceDemoIT` | Load bundled safetensors, solve fixed equation set |
| `TrainExportRoundTripDemoIT` | Train in Java, export safetensors, reload, infer |

Refresh bundled weights after retraining the parent model:

```bash
cp ../java/resources/models/algebra-v1.safetensors src/test/resources/models/
```
