# tensor4j testing

JUnit 5 validates **behavior aligned with tinygrad**, using a mix of hand-specified cases, fixture parity, layout invariants, and finite-difference gradcheck.

## Test layers

| Layer | Purpose | Classes |
|-------|---------|---------|
| **1 — Ops** | Forward math matches expected values | `TensorOpsTest`, `MovementOpsTest` |
| **1c — Float layout** | Flat buffer + strides (not `float[][]`) | `FloatLayoutTest` |
| **1d — Tinygrad parity** | Optional numeric cross-check vs exported fixtures | `TinygradParityTest` |
| **1f — Lazy shape** | Shape metadata before realize (tinygrad `UOp._shape`) | `LazyShapeTest` |
| **1g — Lazy shape parity** | Fixture-driven lazy shape chains | `TinygradLazyShapeParityTest` |
| **1h — Lazy tensor** | Deferred movement ops until `realize()` | `LazyTensorTest` |
| **1i — Lazy compute** | Deferred elementwise / reduce ops | `LazyTensorComputeTest` |
| **1j — Lazy tensor parity** | Fixture-driven lazy tensor chains | `TinygradLazyTensorParityTest` |
| **1k — Lazy DAG** | `LazyUOp` graph / interning / toposort | `LazyGraphTest` |
| **1l — Scheduler** | Fusion + linear kernel schedule (tinygrad `create_schedule` subset) | `LazySchedulerTest`, `LazyScheduleParityTest` |
| **1n — Lazy matmul** | `dot` / `matmul` via broadcast mul + sum (tinygrad decomposition) | `LazyMatmulTest` |
| **1e — All ops** | Every implemented tinygrad-style forward op | `AllOpsForwardTest` |
| **2 — Autograd** | Gradient presence / rules | `AutogradTest`, `MovementAutogradTest` |
| **2l — Lazy autograd** | {@code LazyTensor.backward()} via UOp {@code compute_gradient} rules | `LazyAutogradTest` |
| **2m — Lazy autograd parity** | Lazy vs eager gradient agreement | `LazyAutogradParityTest` |
| **2n — Lazy matmul autograd** | Decomposed dot backward (mul/sum/movement rules) | `LazyMatmulAutogradTest` |
| **2o — Lazy all-ops autograd** | Lazy UOp backward parity vs eager `AllOpsAutogradTest` | `LazyAllOpsAutogradTest` |
| **2p — Lazy gradient rules** | tinygrad `pm_gradient` subset (SUB/DIV/RECIP/MEAN) | `LazyGradientRulesTest` |
| **2b — Gradcheck** | Numeric ∂L/∂x vs autograd on flat buffer | `GradCheckTest` |
| **3 — Domain** | Algebra parsing, training, features | `AlgebraModelTest`, `AlgebraNumericTest`, `MlpAlgebraFitTest` |
| **4 — Integration** | Full training loop | `TrainingIntegrationTest` |
| **IO** | Weight bundles | `ModelLoaderTest`, `SafetensorsTest` |

## Rules

- Use `TensorAssert.assertAllClose(..., eps)` — never raw `assertEquals` on floats
- No Java lambdas in tests (named `*Graph` / `*Action` classes)
- No streams — plain loops in helpers

## Run

```bash
cd sample-apps
npm run test:tensor4j
# or
mvn -f tensor4j/pom.xml test
```

## Algebra demo (standalone)

`algebra-demo/` is a **separate Maven project** (standard `src/test` layout, Failsafe `*IT.java`). Install tensor4j first, then verify the demo module:

```bash
cd tensor4j
mvn install -DskipTests
cd algebra-demo
mvn verify
```

See [algebra-demo/README.md](algebra-demo/README.md).

## Numeric parity with tinygrad (float layout)

tinygrad stores tensors as a **single row-major float buffer** plus shape/strides metadata — same model as tensor4j `StorageBuffer` + `TensorLayout`.

### Primary strategy: Java float reference (cross-platform)

Layer 1/2 tests in `AllOpsForwardTest` and `AllOpsAutogradTest` use **hand-computed Java `float` expectations** via `OpCatalog`. Same results on every JVM; tinygrad may differ slightly at ~1e-5 due to Python/NumPy accumulation order.

### Optional: tinygrad fixture cross-check

1. **`FloatLayoutTest`** (no Python required)  
   Verifies stride indexing, transpose/expand views, and `toFlatArray()` row-major order. Fails if code regresses to nested-array thinking.

2. **`TinygradParityTest`** (fixture-driven)  
   Loads `java/resources/parity/tinygrad-ops.json` and runs the same ops in Java. Expected `data[]` arrays are tinygrad's flattened `numpy()` output (C-order).

3. **`GradCheckTest`** (autograd vs finite differences)  
   Perturbs each element of the flat parameter buffer ±ε and compares to autograd gradients.

### Regenerating fixtures from tinygrad

```bash
cd sample-apps/tensor4j
git clone --depth 1 https://github.com/tinygrad/tinygrad.git vendor/tinygrad   # once
pip install numpy
# Windows PowerShell:
$env:PYTHONPATH = "vendor/tinygrad"
python tools/export_parity_fixtures.py
# Writes java/resources/parity/tinygrad-ops.json
mvn test
```

If Python/tinygrad is unavailable, committed JSON fixtures still run in CI.

### Lazy shape (before realize)

tinygrad tensors are lazy **UOp DAGs**: each `Tensor` holds one root `uop` with `src` parent tuple, hash-consed via `ucache`. `toposort()` walks dependencies; `realize()` / `backward()` use that order.

tensor4j mirrors this in `core/lazy/`:

| tinygrad | tensor4j lazy |
|----------|---------------|
| `UOp(op, src, arg)` | `LazyUOp` with `src[]` parents |
| `UOpMetaClass.ucache` | `LazyUOpCache.intern` |
| `UOp.toposort()` | `LazyGraph.toposort` |
| `Tensor.uop` | `LazyTensor.uop()` |
| Shared subgraph | Same `LazyUOp` reference reused |

| tinygrad | tensor4j lazy | eager tensor4j |
|----------|---------------|----------------|
| {@code UOp._shape} | {@code LazyShape.shape()} | {@code Tensor.shape()} after op |
| {@code reshape/permute/expand} graph nodes | chained {@code LazyShape} ops | {@code TensorLayout} views |
| {@code realize()} | {@code LazyShape.materialize(tensor)} / {@code LazyTensor.realize()} | immediate on op call |
| {@code backward()} | {@code LazyTensor.backward()} → realize + {@code AutogradEngine} | {@code Tensor.backward()} |
| {@code _broadcast_shape} | {@code LazyShape.broadcastShape} | {@code requireSameShape} (no broadcast yet) |

Regenerate lazy fixtures: same {@code export_parity_fixtures.py} run also writes {@code tinygrad-lazy-shapes.json} and {@code tinygrad-lazy-tensor.json}.

### Scheduler and fusion (before execute)

tinygrad lowers the UOp DAG to a **linear kernel list** via `create_schedule` / `run_linear`, fusing elementwise ops into shared kernels when safe.

tensor4j teaching subset in `core/lazy/`:

| tinygrad | tensor4j lazy |
|----------|---------------|
| `get_kernel_graph` / fusion rewrites | `LazyFusion.fuseElementwise` |
| `create_schedule` (RAW/WAR deps) | `LazySchedule.build` |
| `CALL` / fused kernel | `LazyKernel` (`SINGLE` or `FUSED`) |
| `run_linear` | `LazyScheduleExecutor.execute` |
| Fused elementwise eval | `LazyKernelMath.evalFusedBody` |

Non-grad `LazyTensor.realize()` uses the schedule path; autograd builds a **backward UOp DAG** via `LazyGradient` (tinygrad `compute_gradient` subset) and materializes leaf `.grad` on demand.

### Lazy matmul and UOp autograd

| tinygrad | tensor4j lazy |
|----------|---------------|
| `Tensor.dot` → reshape + mul + sum | `LazyDot.dot` / `LazyTensor.matmul` |
| `compute_gradient` / `pm_gradient` | `LazyGradient.compute` |
| `Ops.SUB` / `Ops.DIV` / reciprocal | `SUB`, `DIV`, `RECIP` + movement/reduce rules |
| `Tensor.backward()` | `LazyAutograd.backward` → realize grad UOps |
| Matmul backward | Falls out of mul + sum_axis + movement rules (no `MatMulFunction` on lazy path) |

### What parity checks

| Check | tinygrad | tensor4j |
|-------|----------|----------|
| Flat order | `t.numpy().reshape(-1)` | `tensor.toFlatArray()` |
| Reshape | `Ops.RESHAPE` view | `Tensor.reshape` view |
| Broadcast | stride `0` on expanded axis | `Tensor.expand` |
| Permute | stride swap | `Tensor.permute` |
| Matmul | `@` on row-major | `Tensor.matmul` |

**Tolerance:** `1e-5` for forward parity, `~5e-2` for gradcheck (float32 finite differences).

## Layout

```
java/test/com/github/tensor4j/
  support/TensorAssert.java       # assert_close
  support/ParityFixtureLoader.java
  support/GradCheck.java
  core/TensorOpsTest.java         # layer 1
  core/MovementOpsTest.java
  core/FloatLayoutTest.java       # layer 1c
  core/FloatLayoutTest.java       # layer 1c
  core/TinygradParityTest.java    # layer 1d
  core/lazy/LazyShapeTest.java    # layer 1f
  core/lazy/TinygradLazyShapeParityTest.java  # layer 1g
  core/lazy/LazyTensorTest.java   # layer 1h
  core/lazy/LazyTensorComputeTest.java  # layer 1i
  core/lazy/TinygradLazyTensorParityTest.java  # layer 1j
  core/lazy/LazyGraphTest.java        # layer 1k
  core/lazy/LazySchedulerTest.java  # layer 1l
  core/lazy/LazyScheduleParityTest.java  # layer 1l
  core/lazy/LazyAutogradParityTest.java # layer 2m
  support/LazyShapeFixtureLoaderTest.java
  autograd/AutogradTest.java      # layer 2
  autograd/MovementAutogradTest.java
  autograd/GradCheckTest.java     # layer 2b
  models/algebra/                 # layer 3
  integration/                    # layer 4
  io/ModelLoaderTest.java
  io/SafetensorsTest.java
```

## Weight formats (tinygrad track)

Bundled algebra weights ship as `algebra-v1.safetensors` (default). Legacy **t4j_json** remains available via extension or `-Dtensor4j.weight.format=t4j_json`.

- CLI: `tensor4j export --out model.safetensors` (default) or `--format t4j_json`
- JVM: `-Dtensor4j.weight.format=t4j_json` to load bundled JSON instead of safetensors
- Convert JSON → safetensors: `java -cp ... com.github.tensor4j.tools.ConvertWeights java/resources/models/algebra-v1.t4j.json`

```
java/resources/parity/
  tinygrad-ops.json               # regenerate via tools/export_parity_fixtures.py

tools/
  export_parity_fixtures.py
  export_algebra_model.py         # --format safetensors for tinygrad safe_save
```

When adding ops: add a tinygrad case to `export_parity_fixtures.py`, regenerate JSON, and add a `FloatLayoutTest` if the op introduces new stride behavior.
