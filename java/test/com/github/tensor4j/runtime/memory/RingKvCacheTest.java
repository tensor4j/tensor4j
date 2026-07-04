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

class RingKvCacheTest {

    @Test
    void evictsOldestWhenFull() {
        RingKvCache cache = new RingKvCache(2, 1, 1);
        cache.append(token(1), token(10));
        cache.append(token(2), token(20));
        assertEquals(2, cache.nKv());
        cache.append(token(3), token(30));
        assertEquals(2, cache.nKv());
        assertEquals(2.0f, cache.keysForHead(0).get(0, 0), 1e-5f);
        assertEquals(3.0f, cache.keysForHead(0).get(1, 0), 1e-5f);
        assertEquals(20.0f, cache.valuesForHead(0).get(0, 0), 1e-5f);
        assertEquals(30.0f, cache.valuesForHead(0).get(1, 0), 1e-5f);
    }

    private static InferTensor token(float value) {
        return InferTensor.vector(value);
    }
}
