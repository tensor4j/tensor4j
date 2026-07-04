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
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.core.TensorLayout;
import org.junit.jupiter.api.Test;

/**
 * Layer 1f: lazy shape graph (tinygrad {@code UOp._shape} before {@code realize()}).
 * Validates shape metadata propagation without buffer allocation.
 */
class LazyShapeTest {

    @Test
    void leafShapeIsAvailableWithoutBuffer() {
        LazyShape lazy = LazyShape.leaf(2, 3);
        assertArrayEquals(new int[] {2, 3}, lazy.shape());
        assertEquals(6, lazy.numel());
        assertEquals(0, lazy.movementDepth());
    }

    @Test
    void reshapeRequiresMatchingNumel() {
        LazyShape base = LazyShape.leaf(2, 3);
        LazyShape reshaped = base.reshape(3, 2);
        assertArrayEquals(new int[] {3, 2}, reshaped.shape());
        assertEquals(6, reshaped.numel());
        assertThrows(IllegalArgumentException.class, () -> base.reshape(2, 2).shape());
    }

    @Test
    void permuteReordersDimensions() {
        LazyShape cube = LazyShape.leaf(2, 3, 4);
        LazyShape permuted = cube.permute(2, 0, 1);
        assertArrayEquals(new int[] {4, 2, 3}, permuted.shape());
        assertThrows(IllegalArgumentException.class, () -> cube.permute(1, 0).shape());
    }

    @Test
    void expandBroadcastsSizeOneAxes() {
        LazyShape row = LazyShape.leaf(1, 3);
        LazyShape expanded = row.expand(4, 3);
        assertArrayEquals(new int[] {4, 3}, expanded.shape());
        assertThrows(IllegalArgumentException.class, () -> LazyShape.leaf(2, 3).expand(4, 3).shape());
    }

    @Test
    void reduceAxisCollapsesRequestedDimension() {
        LazyShape matrix = LazyShape.leaf(2, 3);
        LazyShape reduced = matrix.reduceAxis(0);
        assertArrayEquals(new int[] {3}, reduced.shape());
    }

    @Test
    void broadcastShapeMatchesTinygradElementwiseRule() {
        assertArrayEquals(new int[] {3, 4}, LazyShape.broadcastShape(new int[] {3, 1}, new int[] {1, 4}));
        assertArrayEquals(new int[] {2, 3, 4}, LazyShape.broadcastShape(new int[] {3, 4}, new int[] {2, 1, 4}));
        assertThrows(IllegalArgumentException.class,
                () -> LazyShape.broadcastShape(new int[] {2, 3}, new int[] {2, 4}));
    }

    @Test
    void deepMovementChainInfersShapeWithoutRealize() {
        LazyShape lazy = LazyShape.leaf(2, 3)
                .reshape(6)
                .reshape(2, 3)
                .permute(1, 0)
                .expand(3, 2);
        assertArrayEquals(new int[] {3, 2}, lazy.shape());
        assertEquals(4, lazy.movementDepth());
    }

    @Test
    void lazyShapeMatchesEagerTensorAfterMaterialize() {
        float[] data = new float[] {1f, 2f, 3f, 4f, 5f, 6f};
        Tensor root = Tensor.of(data, 2, 3);
        LazyShape lazy = LazyShape.leaf(2, 3).permute(1, 0);
        Tensor eager = lazy.materialize(root);
        assertArrayEquals(lazy.shape(), eager.shape().dims());
        assertAllClose(new float[] {1f, 4f, 2f, 5f, 3f, 6f}, eager);
    }

    @Test
    void expandThenPermuteMaterializesSameAsEagerPath() {
        float[] data = new float[] {1f, 2f, 3f};
        Tensor root = Tensor.of(data, 1, 3);
        LazyShape lazy = LazyShape.leaf(1, 3).expand(2, 3).permute(1, 0);
        Tensor eager = lazy.materialize(root);
        assertArrayEquals(new int[] {3, 2}, lazy.shape());
        assertAllClose(new float[] {1f, 1f, 2f, 2f, 3f, 3f}, eager);
    }

    @Test
    void toLayoutAppliesMovementMetadataToExistingBuffer() {
        Tensor root = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2);
        LazyShape lazy = LazyShape.leaf(2, 2).permute(1, 0);
        TensorLayout layout = lazy.toLayout(root.layout());
        assertArrayEquals(new int[] {1, 2}, layout.strides());
        assertArrayEquals(lazy.shape(), layout.shape());
    }

    @Test
    void reshapeRejectNegativeDimensionsLikeTinygrad() {
        LazyShape base = LazyShape.leaf(2, 3);
        assertThrows(IllegalArgumentException.class, () -> base.reshape(-1, 6).shape());
    }
}
