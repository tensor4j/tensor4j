/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.tensor4j.runtime.infer.InferTensor;
import org.junit.jupiter.api.Test;

/** Ring KV storage — llama.cpp bounded {@code n_ctx} window semantics. */
class RingKvCacheTest {

    private static final int HEAD_DIM = 2;
    private static final int N_HEAD_KV = 1;
    private static final int ROW_WIDTH = N_HEAD_KV * HEAD_DIM;

    @Test
    void appendIncrementsCountUntilMaxSeq() {
        RingKvCache cache = new RingKvCache(4, N_HEAD_KV, HEAD_DIM);
        assertEquals(0, cache.nKv());
        appendToken(cache, 1f, 2f, 3f, 4f);
        assertEquals(1, cache.nKv());
        appendBlock(cache, 2);
        assertEquals(3, cache.nKv());
    }

    @Test
    void ringEvictsOldestWhenFull() {
        RingKvCache cache = new RingKvCache(4, N_HEAD_KV, HEAD_DIM);
        for (int i = 0; i < 6; i++) {
            appendToken(cache, i, i + 1, i + 2, i + 3);
        }
        assertEquals(4, cache.nKv(), "logical n_kv must cap at n_ctx");
        assertEquals(4, cache.maxSeq());
    }

    @Test
    void clearResetsLogicalCount() {
        RingKvCache cache = new RingKvCache(8, N_HEAD_KV, HEAD_DIM);
        appendBlock(cache, 5);
        cache.clear();
        assertEquals(0, cache.nKv());
    }

    @Test
    void gatherHeadReturnsOldestToNewestOrder() {
        RingKvCache cache = new RingKvCache(3, N_HEAD_KV, HEAD_DIM);
        appendToken(cache, 10f, 11f, 12f, 13f);
        appendToken(cache, 20f, 21f, 22f, 23f);
        appendToken(cache, 30f, 31f, 32f, 33f);
        appendToken(cache, 40f, 41f, 42f, 43f);
        assertEquals(3, cache.nKv());
        float[] head0 = cache.keysForHead(0).data();
        assertEquals(20f, head0[0], 1e-6f, "oldest retained slot after eviction");
        assertEquals(40f, head0[4], 1e-6f, "newest slot");
    }

    private static void appendBlock(RingKvCache cache, int rows) {
        float[] k = new float[rows * ROW_WIDTH];
        float[] v = new float[rows * ROW_WIDTH];
        for (int i = 0; i < k.length; i++) {
            k[i] = i;
            v[i] = i + 100;
        }
        cache.appendBlock(InferTensor.of(k, rows, ROW_WIDTH), InferTensor.of(v, rows, ROW_WIDTH));
    }

    private static void appendToken(RingKvCache cache, float k0, float k1, float v0, float v1) {
        cache.append(
                InferTensor.of(new float[] {k0, k1}, 1, ROW_WIDTH),
                InferTensor.of(new float[] {v0, v1}, 1, ROW_WIDTH));
    }
}
