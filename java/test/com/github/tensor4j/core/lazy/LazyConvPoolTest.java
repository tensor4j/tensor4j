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
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.core.ConvTranspose2dArg;
import com.github.tensor4j.core.Pool2dArg;
import com.github.tensor4j.core.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Lazy depthwise conv, conv_transpose2d, and pool2d. */
class LazyConvPoolTest {

    private static final float EPS = 1e-3f;

    @BeforeEach
    void clearCache() {
        LazyGraph.clearInternCacheForTests();
    }

    @Test
    void maxPoolLazyMatchesEager() {
        float[] data = new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 0f, 1f, 2f, 3f, 4f, 5f, 6f};
        Tensor eager = Tensor.of(data, 1, 1, 4, 4).maxPool2d(2, 2);
        LazyTensor lazy = LazyTensor.of(data, 1, 1, 4, 4).maxPool2d(2, 2);
        assertAllClose(eager, lazy.realize(), EPS);
    }

    @Test
    void maxPoolUsesIm2colDecomposition() {
        LazyTensor x = LazyTensor.of(new float[] {1f, 2f, 3f, 4f}, 1, 1, 2, 2);
        assertTrue(hasKind(x.maxPool2d(2, 1).uop(), LazyUOp.Kind.IM2COL));
        assertTrue(hasKind(x.maxPool2d(2, 1).uop(), LazyUOp.Kind.MAX_AXIS));
    }

    @Test
    void avgPoolLazyMatchesEager() {
        float[] data = new float[] {1f, 2f, 3f, 4f};
        assertAllClose(Tensor.of(data, 1, 1, 2, 2).avgPool2d(2, 2),
                LazyTensor.of(data, 1, 1, 2, 2).avgPool2d(2, 2).realize(), EPS);
    }

    @Test
    void depthwiseLazyMatchesEager() {
        float[] xData = new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        float[] wData = new float[] {1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f};
        Tensor eager = Tensor.of(xData, 1, 2, 2, 2).depthwiseConv2d(Tensor.of(wData, 2, 1, 2, 2));
        LazyTensor lazy = LazyTensor.of(xData, 1, 2, 2, 2)
                .depthwiseConv2d(LazyTensor.of(wData, 2, 1, 2, 2));
        assertAllClose(eager, lazy.realize(), EPS);
    }

    @Test
    void convTransposeLazyMatchesEager() {
        float[] xData = new float[] {1f, 2f, 3f, 4f};
        float[] wData = new float[] {1f, 1f, 1f, 1f};
        int[] arg = ConvTranspose2dArg.packed(2, 0, 0, 0, 0, 1);
        Tensor eager = Tensor.of(xData, 1, 1, 2, 2).convTranspose2d(Tensor.of(wData, 1, 1, 2, 2), arg);
        LazyTensor lazy = LazyTensor.of(xData, 1, 1, 2, 2)
                .convTranspose2d(LazyTensor.of(wData, 1, 1, 2, 2), arg);
        assertAllClose(eager, lazy.realize(), EPS);
    }

    @Test
    void convTransposeBuildsUOp() {
        LazyTensor x = LazyTensor.of(new float[] {1f}, 1, 1, 1, 1);
        LazyTensor w = LazyTensor.of(new float[] {1f}, 1, 1, 1, 1);
        assertTrue(hasKind(x.convTranspose2d(w).uop(), LazyUOp.Kind.CONV_TRANSPOSE2D));
    }

    @Test
    void poolBackwardLazyMatchesEager() {
        float[] data = new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 0f, 1f, 2f, 3f, 4f, 5f, 6f};
        Tensor eager = Tensor.of(data, 1, 1, 4, 4).withGrad(true);
        eager.maxPool2d(2, 2).sum().backward();
        LazyTensor lazy = LazyTensor.of(data, 1, 1, 4, 4).withGrad(true);
        lazy.maxPool2d(2, 2).sum().backward();
        assertAllClose(eager.grad(), lazy.grad(), EPS);
    }

    @Test
    void convTransposeBackwardLazyMatchesEager() {
        float[] xData = new float[] {1f, 2f, 3f, 4f};
        float[] wData = new float[] {1f, 0f, 0f, 1f};
        Tensor eagerIn = Tensor.of(xData, 1, 1, 2, 2).withGrad(true);
        Tensor eagerW = Tensor.of(wData, 1, 1, 2, 2).withGrad(true);
        eagerIn.convTranspose2d(eagerW).sum().backward();
        LazyTensor lazyIn = LazyTensor.of(xData, 1, 1, 2, 2).withGrad(true);
        LazyTensor lazyW = LazyTensor.of(wData, 1, 1, 2, 2).withGrad(true);
        lazyIn.convTranspose2d(lazyW).sum().backward();
        assertAllClose(eagerIn.grad(), lazyIn.grad(), EPS);
        assertAllClose(eagerW.grad(), lazyW.grad(), EPS);
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
