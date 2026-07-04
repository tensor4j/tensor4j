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
import static com.github.tensor4j.support.TensorAssert.assertGradPresent;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.core.Conv2dMath;
import com.github.tensor4j.core.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Layer 2s: conv2d + softmax (tinygrad {@code Tensor.conv2d} / {@code Tensor._softmax} subset).
 */
class LazyConvSoftmaxTest {

    private static final float EPS = 1e-4f;

    @BeforeEach
    void clearCache() {
        LazyGraph.clearInternCacheForTests();
    }

    @Test
    void conv2dMatchesTinygradDocExample() {
        float[] input = new float[9];
        for (int i = 0; i < input.length; i++) {
            input[i] = i;
        }
        LazyTensor x = LazyTensor.of(input, 1, 1, 3, 3);
        LazyTensor w = LazyTensor.of(new float[] {1f, 1f, 1f, 1f}, 1, 1, 2, 2);
        assertAllClose(new float[] {8f, 12f, 20f, 24f}, x.conv2d(w).realize(), EPS);
    }

    @Test
    void conv2dLazyMatchesEager() {
        Tensor eagerIn = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 1, 1, 2, 2).withGrad(true);
        Tensor eagerW = Tensor.of(new float[] {1f, 0f, 0f, 1f}, 1, 1, 2, 2).withGrad(true);
        Tensor eagerOut = eagerIn.conv2d(eagerW);

        LazyTensor lazyIn = LazyTensor.wrap(eagerIn);
        LazyTensor lazyW = LazyTensor.wrap(eagerW);
        assertAllClose(eagerOut, lazyIn.conv2d(lazyW).realize(), EPS);
    }

    @Test
    void softmaxMatchesEagerAndSumsToOne() {
        float[] data = new float[] {-2f, -3f, -2f, -1f, 0f, 6f};
        Tensor eager = Tensor.of(data, 2, 3);
        LazyTensor lazy = LazyTensor.of(data, 2, 3);
        Tensor lazyOut = lazy.softmax(-1).realize();
        assertAllClose(eager.softmax(-1), lazyOut, EPS);
        float rowSum = 0f;
        for (int i = 0; i < 3; i++) {
            rowSum += lazyOut.data()[i];
        }
        assertEquals(1f, rowSum, EPS);
    }

    @Test
    void conv2dBackwardMatchesEager() {
        Tensor eagerIn = Tensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f}, 1, 1, 3, 3).withGrad(true);
        Tensor eagerW = Tensor.of(new float[] {1f, 1f, 1f, 1f}, 1, 1, 2, 2).withGrad(true);
        eagerIn.conv2d(eagerW).sum().backward();

        LazyTensor lazyIn = LazyTensor.of(eagerIn.data(), 1, 1, 3, 3).withGrad(true);
        LazyTensor lazyW = LazyTensor.of(eagerW.data(), 1, 1, 2, 2).withGrad(true);
        lazyIn.conv2d(lazyW).sum().backward();

        assertAllClose(eagerIn.grad(), lazyIn.grad(), EPS);
        assertAllClose(eagerW.grad(), lazyW.grad(), EPS);
    }

    @Test
    void softmaxBackwardMatchesEager() {
        float[] data = new float[] {1f, 2f, 3f, 4f};
        Tensor eager = Tensor.of(data, 2, 2).withGrad(true);
        eager.softmax(-1).sum().backward();

        LazyTensor lazy = LazyTensor.of(data, 2, 2).withGrad(true);
        lazy.softmax(-1).sum().backward();

        assertAllClose(eager.grad(), lazy.grad(), EPS);
    }

    @Test
    void softmaxGraphUsesExpDecomposition() {
        LazyTensor x = LazyTensor.of(new float[] {1f, 2f, 3f}, 3);
        assertTrue(hasKind(x.softmax(-1).uop(), LazyUOp.Kind.EXP));
    }

    @Test
    void conv2dBuildsConvUOp() {
        LazyTensor x = LazyTensor.of(new float[4], 1, 1, 2, 2);
        LazyTensor w = LazyTensor.of(new float[4], 1, 1, 2, 2);
        assertTrue(hasKind(x.conv2d(w).uop(), LazyUOp.Kind.IM2COL));
    }

    @Test
    void conv2dGradTargetsPresent() {
        LazyTensor x = LazyTensor.of(new float[] {1f, 2f, 3f, 4f}, 1, 1, 2, 2).withGrad(true);
        LazyTensor w = LazyTensor.of(new float[] {1f, 0f, 0f, 1f}, 1, 1, 2, 2).withGrad(true);
        x.conv2d(w).sum().backward();
        assertGradPresent(x.leafTensor());
        assertGradPresent(w.leafTensor());
    }

    private static boolean hasKind(LazyUOp root, LazyUOp.Kind kind) {
        for (LazyUOp node : LazyGraph.toposort(root)) {
            if (node.op() == kind) {
                return true;
            }
        }
        return false;
    }
}
