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
import static com.github.tensor4j.support.TensorAssert.assertClose;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.Test;

/** Layer 1i: lazy elementwise / reduce ops materialize to match eager math. */
class LazyTensorComputeTest {

    @Test
    void lazyAddBroadcastsOnRealize() {
        LazyTensor left = LazyTensor.of(new float[] {1f, 2f, 3f}, 3, 1);
        LazyTensor right = LazyTensor.of(new float[] {10f, 20f, 30f, 40f}, 1, 4);
        LazyTensor lazy = left.add(right);
        assertFalse(lazy.isRealized());
        assertArrayEquals(new int[] {3, 4}, lazy.shape());
        assertAllClose(new float[] {
            11f, 21f, 31f, 41f,
            12f, 22f, 32f, 42f,
            13f, 23f, 33f, 43f
        }, lazy.realize(), 1e-6f);
    }

    @Test
    void lazyMulThenReluChain() {
        LazyTensor lazy = LazyTensor.of(new float[] {-2f, -1f, 0f, 1f, 2f}, 5)
                .mul(LazyTensor.of(new float[] {2f, 2f, 2f, 2f, 2f}, 5))
                .relu();
        assertAllClose(new float[] {0f, 0f, 0f, 2f, 4f}, lazy.realize(), 1e-6f);
    }

    @Test
    void lazySumReducesAllElements() {
        LazyTensor lazy = LazyTensor.of(new float[] {1f, 2f, 3f, 4f}, 4).sum();
        assertArrayEquals(new int[] {1}, lazy.shape());
        assertClose(10f, lazy.realize().data()[0], 1e-6f);
    }

    @Test
    void lazySumAxisMatchesEager() {
        float[] data = new float[] {1f, 2f, 3f, 4f, 5f, 6f};
        LazyTensor lazy = LazyTensor.of(data, 2, 3).sumAxis(0);
        assertArrayEquals(new int[] {3}, lazy.shape());
        assertAllClose(new float[] {5f, 7f, 9f}, lazy.realize(), 1e-6f);
    }

    @Test
    void movementThenAddFusesAtRealize() {
        LazyTensor left = LazyTensor.of(new float[] {1f, 2f, 3f}, 3).expand(2, 3);
        LazyTensor right = LazyTensor.of(new float[] {10f, 20f, 30f}, 3).expand(2, 3);
        LazyTensor lazy = left.add(right);
        assertAllClose(new float[] {
            11f, 22f, 33f,
            11f, 22f, 33f
        }, lazy.realize(), 1e-6f);
    }
}
