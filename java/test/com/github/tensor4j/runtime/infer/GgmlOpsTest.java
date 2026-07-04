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

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.tensor4j.support.TensorAssert;
import org.junit.jupiter.api.Test;

class GgmlOpsTest {

    @Test
    void mulMat2x2() {
        InferTensor a = InferTensor.matrix(2, 2, 1, 2, 3, 4);
        InferTensor b = InferTensor.matrix(2, 2, 5, 6, 7, 8);
        InferTensor c = GgmlOps.mulMat(a, b);
        assertEquals(19.0f, c.get(0, 0), 1e-5f);
        assertEquals(22.0f, c.get(0, 1), 1e-5f);
        assertEquals(43.0f, c.get(1, 0), 1e-5f);
        assertEquals(50.0f, c.get(1, 1), 1e-5f);
    }

    @Test
    void rmsNormSingleRow() {
        InferTensor x = InferTensor.vector(3, 4);
        InferTensor weight = InferTensor.vector(1, 1);
        InferTensor y = GgmlOps.rmsNorm(x, weight, 1e-5f);
        float scale = (float) (1.0 / Math.sqrt((9 + 16) / 2.0 + 1e-5));
        TensorAssert.assertAllClose(new float[] {3 * scale, 4 * scale}, y.data(), 1e-4f);
    }

    @Test
    void siluKnownValue() {
        InferTensor x = InferTensor.vector(0.0f);
        assertEquals(0.0f, GgmlOps.silu(x).get(0, 0), 1e-6f);
        InferTensor one = InferTensor.vector(1.0f);
        assertEquals(0.731058f, GgmlOps.silu(one).get(0, 0), 1e-4f);
    }

    @Test
    void swigluMatchesSiluTimesUp() {
        InferTensor gate = InferTensor.vector(1, 2);
        InferTensor up = InferTensor.vector(3, 4);
        TensorAssert.assertAllClose(
                GgmlOps.mul(GgmlOps.silu(gate), up).data(),
                GgmlOps.swiglu(gate, up).data(),
                1e-6f);
    }

    @Test
    void qkScoresAndContext() {
        InferTensor q = InferTensor.matrix(1, 2, 1, 0);
        InferTensor k = InferTensor.matrix(2, 2, 1, 0, 0, 1);
        InferTensor scores = GgmlOps.qkScores(q, k);
        assertEquals(1.0f, scores.get(0, 0), 1e-5f);
        assertEquals(0.0f, scores.get(0, 1), 1e-5f);
        InferTensor probs = GgmlOps.softmaxRows(scores);
        assertEquals(1.0f, probs.get(0, 0) + probs.get(0, 1), 1e-5f);
        InferTensor ctx = GgmlOps.attnContext(probs, k);
        assertEquals(0.731058f, ctx.get(0, 0), 1e-3f);
        assertEquals(0.268942f, ctx.get(0, 1), 1e-3f);
    }
}
