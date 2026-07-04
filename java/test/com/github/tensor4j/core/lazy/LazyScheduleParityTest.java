/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.core.lazy;

import static com.github.tensor4j.support.TensorAssert.assertAllClose;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Layer 1m: fused schedule parity vs unfused eager realize path. */
class LazyScheduleParityTest {

    @BeforeEach
    void clearCache() {
        LazyGraph.clearInternCacheForTests();
    }

    @Test
    void broadcastAddReluMatchesRealize() {
        LazyTensor left = LazyTensor.of(new float[] {1f, 2f, 3f}, 3, 1);
        LazyTensor right = LazyTensor.of(new float[] {10f, 20f, 30f}, 1, 3);
        LazyTensor lazy = left.expand(3, 3).add(right.expand(3, 3)).relu();
        assertAllClose(lazy.realize(), lazy.schedule().execute(), 1e-6f);
    }

    @Test
    void deepUnaryChainMatchesRealize() {
        LazyTensor lazy = LazyTensor.of(new float[] {-1f, 2f, -3f, 4f}, 4).neg().relu().neg();
        assertAllClose(lazy.realize(), lazy.schedule().execute(), 1e-6f);
    }

    @Test
    void permuteThenReluMatchesRealize() {
        float[] data = new float[] {1f, 2f, 3f, 4f};
        LazyTensor lazy = LazyTensor.of(data, 2, 2).permute(1, 0).relu();
        assertAllClose(lazy.realize(), lazy.schedule().execute(), 1e-6f);
    }
}
