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

import com.github.tensor4j.core.Conv2dArg;
import com.github.tensor4j.core.Conv2dMath;
import com.github.tensor4j.core.ConvIm2Col;
import com.github.tensor4j.core.ConvWinograd;
import com.github.tensor4j.core.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Extended conv2d + log_softmax (Winograd, im2col, dilation, groups). */
class LazyConvSoftmaxExtendedTest {

    private static final float EPS = 1e-3f;

    @BeforeEach
    void clearCache() {
        LazyGraph.clearInternCacheForTests();
    }

    @Test
    void im2colMatchesDirectForward() {
        Tensor x = Tensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f}, 1, 1, 3, 3);
        Tensor w = Tensor.of(new float[] {1f, 1f, 1f, 1f}, 1, 1, 2, 2);
        int[] arg = Conv2dMath.defaultArg();
        assertAllClose(Conv2dMath.forwardIm2Col(x, w, arg), ConvIm2Col.forward(x, w, Conv2dArg.parse(arg)), EPS);
    }

    @Test
    void winogradMatchesIm2colFor3x3() {
        Tensor x = Tensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 0f, 1f, 2f, 3f, 4f, 5f, 6f}, 1, 1, 4, 4);
        Tensor w = Tensor.of(new float[] {1f, 0f, 1f, 0f, 1f, 0f, 1f, 0f, 1f}, 1, 1, 3, 3);
        int[] arg = Conv2dMath.defaultArg();
        assertAllClose(ConvWinograd.forward(x, w, Conv2dArg.parse(arg)),
                ConvIm2Col.forward(x, w, Conv2dArg.parse(arg)), EPS);
    }

    @Test
    void conv2dUsesWinogradFor3x3Dispatch() {
        Tensor x = Tensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f}, 1, 1, 3, 3);
        Tensor w = Tensor.of(new float[] {1f, 0f, 1f, 0f, 1f, 0f, 1f, 0f, 1f}, 1, 1, 3, 3);
        assertAllClose(Conv2dMath.forwardWinograd(x, w, Conv2dMath.defaultArg()),
                Conv2dMath.forward(x, w, Conv2dMath.defaultArg()), EPS);
    }

    @Test
    void dilationMatchesIm2col() {
        float[] data = new float[16];
        for (int i = 0; i < data.length; i++) {
            data[i] = i + 1f;
        }
        Tensor x = Tensor.of(data, 1, 1, 4, 4);
        Tensor w = Tensor.of(new float[] {1f, 1f, 1f, 1f}, 1, 1, 2, 2);
        int[] arg = Conv2dArg.packed(1, 0, 0, 0, 0, 1, 2);
        assertAllClose(ConvIm2Col.forward(x, w, Conv2dArg.parse(arg)),
                Conv2dMath.forward(x, w, arg), EPS);
    }

    @Test
    void groupedConvForward() {
        Tensor x = Tensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, 1, 2, 2, 2);
        Tensor w = Tensor.of(new float[] {1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f}, 2, 1, 2, 2);
        int[] arg = Conv2dArg.packed(1, 0, 0, 0, 0, 2);
        Tensor out = Conv2dMath.forward(x, w, arg);
        assertAllClose(new float[] {10f, 26f}, out, EPS);
    }

    @Test
    void lazyConvUsesIm2colDecomposition() {
        LazyTensor x = LazyTensor.of(new float[] {1f, 2f, 3f, 4f}, 1, 1, 2, 2);
        LazyTensor w = LazyTensor.of(new float[] {1f, 0f, 0f, 1f}, 1, 1, 2, 2);
        assertTrue(hasKind(x.conv2d(w).uop(), LazyUOp.Kind.IM2COL));
        assertTrue(hasKind(x.conv2d(w).uop(), LazyUOp.Kind.PAD));
    }

    @Test
    void logSoftmaxMatchesEager() {
        float[] data = new float[] {1f, 2f, 3f, 4f};
        Tensor eager = Tensor.of(data, 2, 2);
        LazyTensor lazy = LazyTensor.of(data, 2, 2);
        assertAllClose(eager.logSoftmax(-1), lazy.logSoftmax(-1).realize(), EPS);
    }

    @Test
    void logSoftmaxBackwardMatchesEager() {
        float[] data = new float[] {1f, 2f, 3f, 4f};
        Tensor eager = Tensor.of(data, 2, 2).withGrad(true);
        eager.logSoftmax(-1).sum().backward();
        LazyTensor lazy = LazyTensor.of(data, 2, 2).withGrad(true);
        lazy.logSoftmax(-1).sum().backward();
        assertAllClose(eager.grad(), lazy.grad(), EPS);
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
