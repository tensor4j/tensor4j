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

import org.junit.jupiter.api.Test;

/** Forward conv2d + softmax (tinygrad doc examples). */
class ConvSoftmaxOpsTest {

    private static final float EPS = 1e-4f;

    @Test
    void conv2dTinygradExample() {
        float[] input = new float[9];
        for (int i = 0; i < input.length; i++) {
            input[i] = i;
        }
        Tensor x = Tensor.of(input, 1, 1, 3, 3);
        Tensor w = Tensor.of(new float[] {1f, 1f, 1f, 1f}, 1, 1, 2, 2);
        assertAllClose(new float[] {8f, 12f, 20f, 24f}, x.conv2d(w), EPS);
    }

    @Test
    void softmaxRowsSumToOne() {
        Tensor out = Tensor.of(new float[] {-2f, -3f, -2f, -1f, 0f, 6f}, 2, 3).softmax(-1);
        for (int row = 0; row < 2; row++) {
            float sum = 0f;
            for (int col = 0; col < 3; col++) {
                sum += out.data()[row * 3 + col];
            }
            assertEquals(1f, sum, EPS);
        }
    }
}
