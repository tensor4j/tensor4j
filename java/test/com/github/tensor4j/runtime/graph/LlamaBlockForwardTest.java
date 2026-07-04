/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.graph;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.tensor4j.runtime.infer.GgmlOps;
import com.github.tensor4j.runtime.infer.InferTensor;
import com.github.tensor4j.runtime.infer.InferWeight;
import com.github.tensor4j.runtime.infer.RopeConfig;
import com.github.tensor4j.runtime.memory.KvCache;
import com.github.tensor4j.support.TensorAssert;
import org.junit.jupiter.api.Test;

class LlamaBlockForwardTest {

    private static final float EPS = 1e-5f;
    private static final RopeConfig NO_ROPE = RopeConfig.disabled();

    @Test
    void tinyBlockForward() {
        int nEmbd = 2;
        KvCache cache = new KvCache(4, 1, nEmbd);
        InferTensor x = InferTensor.vector(1, 2);
        InferTensor ones = InferTensor.vector(1, 1);
        InferTensor identity = identity(nEmbd);
        InferTensor zero = zeroMat(nEmbd, nEmbd);

        LlamaBlockForward.Weights weights = new LlamaBlockForward.Weights(
                eager(ones), eager(identity), eager(identity), eager(identity), eager(identity),
                eager(ones), eager(zero), eager(identity), eager(zero));

        InferTensor out = LlamaBlockForward.forward(x, weights, cache, 1, 1, EPS, null, NO_ROPE);
        assertEquals(1, cache.nKv());

        KvCache verify = new KvCache(4, 1, nEmbd);
        InferTensor attn = LlamaAttentionForward.forward(
                x, ones, identity, identity, identity, identity, verify, 1, 1, EPS, null, NO_ROPE, true);
        InferTensor afterAttn = GgmlOps.add(x, attn);
        InferTensor ffnIn = GgmlOps.rmsNorm(afterAttn, ones, EPS);
        InferTensor ffnMid = GgmlOps.swiglu(GgmlOps.mulMatOut(ffnIn, zero), GgmlOps.mulMatOut(ffnIn, identity));
        InferTensor expected = GgmlOps.add(afterAttn, GgmlOps.mulMatOut(ffnMid, zero));

        TensorAssert.assertAllClose(expected.data(), out.data(), 1e-3f);
    }

    private static InferTensor identity(int n) {
        float[] data = new float[n * n];
        for (int i = 0; i < n; i++) {
            data[i * n + i] = 1.0f;
        }
        return InferTensor.of(data, n, n);
    }

    private static InferWeight eager(InferTensor tensor) {
        return InferWeight.eager(tensor);
    }

    private static InferTensor zeroMat(int rows, int cols) {
        return InferTensor.zeros(rows, cols);
    }
}
