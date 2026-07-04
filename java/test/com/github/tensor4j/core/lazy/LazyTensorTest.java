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
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.core.Tensor;
import org.junit.jupiter.api.Test;

/** Layer 1h: lazy tensor graph — deferred movement ops until {@link LazyTensor#realize()}. */
class LazyTensorTest {

    @Test
    void shapeAvailableBeforeRealize() {
        LazyTensor lazy = LazyTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 2, 3).permute(1, 0);
        assertFalse(lazy.isRealized());
        assertArrayEquals(new int[] {3, 2}, lazy.shape());
        assertEquals(1, lazy.graphDepth());
    }

    @Test
    void realizeIsIdempotentAndCached() {
        LazyTensor lazy = LazyTensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2).reshape(4);
        assertFalse(lazy.isRealized());
        Tensor first = lazy.realize();
        assertTrue(lazy.isRealized());
        Tensor second = lazy.realize();
        assertSame(first, second);
        assertAllClose(new float[] {1f, 2f, 3f, 4f}, first);
    }

    @Test
    void movementChainSharesLeafBufferWithoutCopy() {
        LazyTensor lazy = LazyTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 2, 3).permute(1, 0);
        assertTrue(lazy.isMovementOnly());
        Tensor materialized = lazy.realize();
        assertSame(lazy.leafTensor().buffer().data(), materialized.buffer().data());
        assertAllClose(new float[] {1f, 4f, 2f, 5f, 3f, 6f}, materialized);
    }

    @Test
    void deepMovementChainMatchesEagerTensorPath() {
        float[] data = new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        Tensor eager = Tensor.of(data, 2, 2, 2).permute(2, 0, 1).reshape(8).reshape(2, 2, 2);
        LazyTensor lazy = LazyTensor.of(data, 2, 2, 2).permute(2, 0, 1).reshape(8).reshape(2, 2, 2);
        assertArrayEquals(eager.shape().dims(), lazy.shape());
        assertAllClose(eager.toFlatArray(), lazy.realize(), 1e-6f);
        assertEquals(3, lazy.graphDepth());
    }

    @Test
    void expandViewDeferredUntilRealize() {
        LazyTensor lazy = LazyTensor.of(new float[] {1f, 2f, 3f}, 1, 3).expand(2, 3);
        assertFalse(lazy.isRealized());
        assertArrayEquals(new int[] {2, 3}, lazy.shape());
        assertAllClose(new float[] {1f, 2f, 3f, 1f, 2f, 3f}, lazy.realize());
    }

    @Test
    void movementShapeMatchesLazyShapeHelper() {
        LazyTensor lazy = LazyTensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2).permute(1, 0);
        assertArrayEquals(lazy.movementShape().shape(), lazy.shape());
    }
}
