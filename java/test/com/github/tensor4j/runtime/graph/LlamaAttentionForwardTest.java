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
import com.github.tensor4j.runtime.infer.RopeConfig;
import com.github.tensor4j.runtime.memory.KvCache;
import com.github.tensor4j.support.TensorAssert;
import org.junit.jupiter.api.Test;

class LlamaAttentionForwardTest {

    private static final float EPS = 1e-5f;
    private static final RopeConfig NO_ROPE = RopeConfig.disabled();

    @Test
    void singleHeadIdentityWeights() {
        int nEmbd = 2;
        KvCache cache = new KvCache(4, 1, nEmbd);
        InferTensor x = InferTensor.vector(1, 2);
        InferTensor ones = InferTensor.vector(1, 1);
        InferTensor identity = identity(nEmbd);

        InferTensor out = LlamaAttentionForward.forward(
                x, ones, identity, identity, identity, identity, cache, 1, 1, EPS, null, NO_ROPE, true);

        float scale = (float) (1.0 / Math.sqrt((1 + 4) / 2.0 + EPS));
        float[] expected = {1 * scale, 2 * scale};
        TensorAssert.assertAllClose(expected, out.data(), 1e-3f);
        assertEquals(1, cache.nKv());
    }

    @Test
    void gqaWeightShapes() {
        int nEmbd = 4;
        int nHead = 4;
        int nHeadKv = 2;
        int headDim = nEmbd / nHead;
        int kvWidth = nHeadKv * headDim;
        KvCache cache = new KvCache(4, nHeadKv, headDim);
        InferTensor ones = InferTensor.vector(1, 1, 1, 1);
        InferTensor wq = identity(nEmbd);
        InferTensor wk = identityRows(kvWidth, nEmbd);
        InferTensor wv = identityRows(kvWidth, nEmbd);
        InferTensor wo = identity(nEmbd);

        LlamaAttentionForward.forward(
                InferTensor.vector(0.1f, 0.2f, 0.3f, 0.4f),
                ones, wq, wk, wv, wo, cache, nHead, nHeadKv, EPS, null, NO_ROPE, true);
        assertEquals(1, cache.nKv());
        assertEquals(kvWidth, cache.keys().cols());
    }

    @Test
    void decodeSecondTokenUsesCache() {
        int nEmbd = 2;
        KvCache cache = new KvCache(4, 1, nEmbd);
        InferTensor ones = InferTensor.vector(1, 1);
        InferTensor identity = identity(nEmbd);

        LlamaAttentionForward.forward(
                InferTensor.vector(1, 0), ones, identity, identity, identity, identity,
                cache, 1, 1, EPS, null, NO_ROPE, true);
        assertEquals(1, cache.nKv());

        InferTensor out = LlamaAttentionForward.forward(
                InferTensor.vector(0, 1), ones, identity, identity, identity, identity,
                cache, 1, 1, EPS, null, NO_ROPE, true);
        assertEquals(2, cache.nKv());
        InferTensor q = GgmlOps.rmsNorm(InferTensor.vector(0, 1), ones, EPS);
        InferTensor keys = cache.keysForHead(0);
        InferTensor values = cache.valuesForHead(0);
        float headScale = (float) (1.0 / Math.sqrt(nEmbd));
        InferTensor ctx = GgmlOps.attnContext(
                GgmlOps.softmaxRows(GgmlOps.scale(GgmlOps.qkScores(q, keys), headScale)),
                values);
        TensorAssert.assertAllClose(ctx.data(), out.data(), 1e-3f);
    }

    @Test
    void prefillTwoTokens() {
        int nEmbd = 2;
        KvCache cache = new KvCache(4, 1, nEmbd);
        InferTensor ones = InferTensor.vector(1, 1);
        InferTensor identity = identity(nEmbd);
        InferTensor x = InferTensor.matrix(2, 2, 1, 0, 0, 1);

        InferTensor out = LlamaAttentionForward.forward(
                x, ones, identity, identity, identity, identity, cache, 1, 1, EPS, null, NO_ROPE, true);
        assertEquals(2, cache.nKv());
        assertEquals(2, out.rows());
        float scale0 = (float) (1.0 / Math.sqrt(0.5 + EPS));
        assertEquals(1 * scale0, out.get(0, 0), 1e-3f);
    }

    private static InferTensor identity(int n) {
        float[] data = new float[n * n];
        for (int i = 0; i < n; i++) {
            data[i * n + i] = 1.0f;
        }
        return InferTensor.of(data, n, n);
    }

    private static InferTensor identityRows(int rows, int cols) {
        float[] data = new float[rows * cols];
        int n = Math.min(rows, cols);
        for (int i = 0; i < n; i++) {
            data[i * cols + i] = 1.0f;
        }
        return InferTensor.of(data, rows, cols);
    }
}
