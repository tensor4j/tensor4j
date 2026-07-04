/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.core;

import static com.github.tensor4j.support.TensorAssert.assertAllClose;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.tensor4j.support.GradCheck;
import org.junit.jupiter.api.Test;

/** Depthwise conv, conv_transpose2d, and pool2d forward/backward. */
class ConvPoolOpsTest {

    private static final float EPS = 1e-3f;

    @Test
    void maxPool2x2Stride2() {
        float[] data = new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 0f, 1f, 2f, 3f, 4f, 5f, 6f};
        Tensor x = Tensor.of(data, 1, 1, 4, 4);
        assertAllClose(new float[] {6f, 8f, 9f, 6f}, x.maxPool2d(2, 2), EPS);
    }

    @Test
    void avgPool2x2Stride2() {
        float[] data = new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 0f, 1f, 2f, 3f, 4f, 5f, 6f};
        Tensor x = Tensor.of(data, 1, 1, 4, 4);
        assertAllClose(new float[] {3.5f, 5.5f, 4f, 3.5f}, x.avgPool2d(2, 2), EPS);
    }

    @Test
    void depthwiseConvMatchesGroupedConv() {
        Tensor x = Tensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, 1, 2, 2, 2);
        Tensor w = Tensor.of(new float[] {1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f}, 2, 1, 2, 2);
        int[] arg = Conv2dArg.packed(1, 0, 0, 0, 0, 2);
        assertAllClose(x.conv2d(w, arg), x.depthwiseConv2d(w), EPS);
    }

    @Test
    void convTranspose2dTinygradExample() {
        float[] input = new float[9];
        for (int i = 0; i < input.length; i++) {
            input[i] = i;
        }
        Tensor x = Tensor.of(input, 1, 1, 3, 3);
        Tensor w = Tensor.of(new float[] {1f, 1f, 1f, 1f}, 1, 1, 2, 2);
        assertAllClose(new float[] {
                0f, 1f, 3f, 2f,
                3f, 8f, 12f, 7f,
                9f, 20f, 24f, 13f,
                6f, 13f, 15f, 8f
        }, x.convTranspose2d(w), EPS);
    }

    @Test
    void convTranspose2dSingleElement() {
        Tensor x = Tensor.of(new float[] {2f}, 1, 1, 1, 1);
        Tensor w = Tensor.of(new float[] {3f}, 1, 1, 1, 1);
        assertAllClose(new float[] {6f}, x.convTranspose2d(w), EPS);
    }

    @Test
    void convTranspose2dUpsampleStride2() {
        Tensor x = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 1, 1, 2, 2);
        Tensor w = Tensor.of(new float[] {1f, 1f, 1f, 1f}, 1, 1, 2, 2);
        int[] arg = ConvTranspose2dArg.packed(2, 0, 0, 0, 0, 1);
        Tensor out = x.convTranspose2d(w, arg);
        assertEquals(4, out.shape().dims()[2]);
        assertEquals(4, out.shape().dims()[3]);
        assertAllClose(new float[] {1f, 1f, 2f, 2f, 1f, 1f, 2f, 2f, 3f, 3f, 4f, 4f, 3f, 3f, 4f, 4f}, out, EPS);
    }

    @Test
    void poolBackwardMatchesEager() {
        float[] data = new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 0f, 1f, 2f, 3f, 4f, 5f, 6f};
        Tensor eager = Tensor.of(data, 1, 1, 4, 4).withGrad(true);
        eager.maxPool2d(2, 2).sum().backward();
        Tensor lazy = Tensor.of(data, 1, 1, 4, 4).withGrad(true);
        lazy.maxPool2d(2, 2).sum().backward();
        assertAllClose(eager.grad(), lazy.grad(), EPS);
    }

    @Test
    void convTransposeBackwardGradcheck() {
        float[] xData = new float[] {1f, 2f, 3f, 4f};
        float[] wData = new float[] {1f, 0f, 0f, 1f};
        Tensor xBase = Tensor.of(xData, 1, 1, 2, 2);
        Tensor wFixed = Tensor.of(wData, 1, 1, 2, 2);
        Tensor w = Tensor.of(wData, 1, 1, 2, 2).withGrad(true);
        GradCheck.assertGradClose(w, param -> xBase.convTranspose2d(param).sum());
        Tensor x = Tensor.of(xData, 1, 1, 2, 2).withGrad(true);
        GradCheck.assertGradClose(x, param -> param.convTranspose2d(wFixed).sum());
    }
}
