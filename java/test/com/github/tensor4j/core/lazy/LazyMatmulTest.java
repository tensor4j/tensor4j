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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.core.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Layer 1n: lazy matmul via tinygrad dot decomposition (broadcast mul + sum). */
class LazyMatmulTest {

    @BeforeEach
    void clearCache() {
        LazyGraph.clearInternCacheForTests();
    }

    @Test
    void matmul2x2MatchesEager() {
        float[] aData = new float[] {1f, 2f, 3f, 4f};
        float[] bData = new float[] {5f, 6f, 7f, 8f};
        Tensor eager = Tensor.of(aData, 2, 2).matmul(Tensor.of(bData, 2, 2));

        LazyTensor lazy = LazyTensor.of(aData, 2, 2).matmul(LazyTensor.of(bData, 2, 2));
        assertFalse(lazy.isRealized());
        assertAllClose(eager, lazy.realize(), 1e-6f);
    }

    @Test
    void matmulBuildsMulSumGraphNotDedicatedOp() {
        LazyTensor lazy = LazyTensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2)
                .matmul(LazyTensor.of(new float[] {1f, 0f, 0f, 1f}, 2, 2));
        assertTrue(lazy.graphNodeCount() >= 5);
        assertTrue(LazyGraph.references(lazy.uop(), findOp(lazy, LazyUOp.Kind.MUL)));
        assertTrue(LazyGraph.references(lazy.uop(), findOp(lazy, LazyUOp.Kind.SUM_AXIS)));
    }

    @Test
    void batchedShapeMatmulMatchesEager() {
        float[] aData = new float[] {1f, 2f, 3f, 4f, 5f, 6f};
        float[] bData = new float[] {1f, 0f, 0f, 1f, 2f, 2f, 3f, 3f, 4f, 4f, 5f, 5f};
        Tensor eager = Tensor.of(aData, 2, 3).matmul(Tensor.of(bData, 3, 4));
        LazyTensor lazy = LazyTensor.of(aData, 2, 3).matmul(LazyTensor.of(bData, 3, 4));
        assertArrayEquals(new int[] {2, 4}, lazy.shape());
        assertAllClose(eager, lazy.realize(), 1e-6f);
    }

    private static LazyUOp findOp(LazyTensor lazy, LazyUOp.Kind kind) {
        for (LazyUOp node : lazy.toposort()) {
            if (node.op() == kind) {
                return node;
            }
        }
        throw new AssertionError("missing op " + kind);
    }
}
