/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.infer;

import static com.github.tensor4j.support.TensorAssert.assertAllClose;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import com.github.tensor4j.support.GgmlParityFixtureLoader;
import com.github.tensor4j.support.GgmlParityFixtureLoader.GgmlParityCase;
import com.github.tensor4j.support.ParityFixtureLoader.ParityTensor;
import org.junit.jupiter.api.Test;

/**
 * Per-op ggml infer kernel parity vs tinygrad-exported fixtures
 * ({@code java/resources/parity/tinygrad-ggml-ops.json}).
 *
 * <p>Regenerate: {@code python tools/export_ggml_ops_parity.py}
 */
class TinygradGgmlOpsParityTest {

    private static final String FIXTURE = "/parity/tinygrad-ggml-ops.json";

    @Test
    void ggmlOpsMatchTinygradFixtures() {
        for (GgmlParityCase parityCase : GgmlParityFixtureLoader.loadResource(FIXTURE)) {
            InferTensor actual = runCase(parityCase);
            assertNotNull(actual, "no result for case " + parityCase.name());
            assertAllClose(parityCase.expected().data(), actual.data(), parityCase.tolerance());
        }
    }

    private static InferTensor runCase(GgmlParityCase parityCase) {
        return switch (parityCase.op()) {
            case "ggml_mul_mat" -> GgmlOps.mulMat(tensor(parityCase.inputs().get(0)), tensor(parityCase.inputs().get(1)));
            case "ggml_add" -> GgmlOps.add(tensor(parityCase.inputs().get(0)), tensor(parityCase.inputs().get(1)));
            case "ggml_mul" -> GgmlOps.mul(tensor(parityCase.inputs().get(0)), tensor(parityCase.inputs().get(1)));
            case "ggml_rms_norm" -> GgmlOps.rmsNorm(
                    tensor(parityCase.inputs().get(0)),
                    parityCase.inputs().size() > 1 ? tensor(parityCase.inputs().get(1)) : null,
                    parityCase.eps());
            case "ggml_silu" -> GgmlOps.silu(tensor(parityCase.input()));
            case "ggml_swiglu" -> GgmlOps.swiglu(
                    tensor(parityCase.inputs().get(0)), tensor(parityCase.inputs().get(1)));
            case "ggml_softmax_rows" -> GgmlOps.softmaxRows(tensor(parityCase.input()));
            case "ggml_scale" -> GgmlOps.scale(tensor(parityCase.input()), parityCase.scale());
            case "ggml_qk_scores" -> GgmlOps.qkScores(
                    tensor(parityCase.inputs().get(0)), tensor(parityCase.inputs().get(1)));
            case "ggml_attn_context" -> GgmlOps.attnContext(
                    tensor(parityCase.inputs().get(0)), tensor(parityCase.inputs().get(1)));
            case "ggml_causal_mask" -> GgmlOps.applyCausalMask(tensor(parityCase.input()), parityCase.pastKv());
            case "ggml_rope_heads" -> Rope.applyHeads(
                    tensor(parityCase.input()),
                    parityCase.nHeads(),
                    parityCase.headDim(),
                    parityCase.positions(),
                    toRopeConfig(parityCase));
            default -> throw new IllegalArgumentException("unsupported ggml parity op: " + parityCase.op());
        };
    }

    private static RopeConfig toRopeConfig(GgmlParityCase parityCase) {
        RopeScalingType scaling = switch (parityCase.scaling()) {
            case "yarn" -> RopeScalingType.YARN;
            case "linear" -> RopeScalingType.LINEAR;
            default -> RopeScalingType.NONE;
        };
        return new RopeConfig(
                parityCase.freqBase(),
                parityCase.freqScale(),
                parityCase.ropeDim(),
                scaling,
                parityCase.yarnExtFactor(),
                parityCase.yarnAttnFactor(),
                parityCase.yarnBetaFast(),
                parityCase.yarnBetaSlow(),
                parityCase.yarnOrigCtx());
    }

    private static InferTensor tensor(ParityTensor spec) {
        int rows = spec.shape()[0];
        int cols = spec.shape().length > 1 ? spec.shape()[1] : 1;
        return InferTensor.of(spec.data(), rows, cols);
    }
}
