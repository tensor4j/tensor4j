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

class KvCacheTest {

    @Test
    void appendAndReadHead() {
        KvCache cache = new KvCache(8, 1, 2);
        cache.append(token(1, 2), token(3, 4));
        cache.append(token(5, 6), token(7, 8));
        assertEquals(2, cache.nKv());
        InferTensor k = cache.keysForHead(0);
        assertEquals(1.0f, k.get(0, 0), 1e-5f);
        assertEquals(5.0f, k.get(1, 0), 1e-5f);
        assertEquals(7.0f, cache.valuesForHead(0).get(1, 0), 1e-5f);
        assertEquals(8.0f, cache.valuesForHead(0).get(1, 1), 1e-5f);
    }

    @Test
    void appendBlockPrefill() {
        KvCache cache = new KvCache(8, 2, 1);
        InferTensor kBlock = InferTensor.matrix(2, 2, 1, 2, 3, 4);
        InferTensor vBlock = InferTensor.matrix(2, 2, 5, 6, 7, 8);
        cache.appendBlock(kBlock, vBlock);
        assertEquals(2, cache.nKv());
        assertEquals(4.0f, cache.keysForHead(1).get(1, 0), 1e-5f);
    }

    private static InferTensor token(float k0, float k1) {
        return InferTensor.vector(k0, k1);
    }
}
