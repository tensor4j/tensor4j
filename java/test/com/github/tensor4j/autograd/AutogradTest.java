/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.autograd;

import static com.github.tensor4j.support.TensorAssert.assertAllClose;
import static com.github.tensor4j.support.TensorAssert.assertGradPresent;
import static com.github.tensor4j.support.TensorAssert.matrix2d;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import com.github.tensor4j.core.Tensor;
import org.junit.jupiter.api.Test;

/** Layer 2: autograd / gradient-flow behavior. */
class AutogradTest {

    @Test
    void matmulProducesGradOnInputs() {
        Tensor x = matrix2d(new float[][] {{1f, 2f}}).withGrad(true);
        Tensor w = matrix2d(new float[][] {{3f}, {4f}}).withGrad(true);
        Tensor y = x.matmul(w);
        Tensor loss = y.mul(y).mean();
        loss.backward();
        assertGradPresent(x);
        assertGradPresent(w);
    }

    @Test
    void reluBlocksNegativeInputGradient() {
        Tensor x = Tensor.of(new float[] {-1f, 2f}, 2).withGrad(true);
        Tensor y = x.relu();
        Tensor loss = y.mean();
        loss.backward();
        assertNotNull(x.grad());
        assertAllClose(new float[] {0f, 0.5f}, x.grad());
    }

    @Test
    void chainThroughLinearOps() {
        Tensor a = Tensor.of(new float[] {2f, 3f}, 1, 2).withGrad(true);
        Tensor b = Tensor.of(new float[] {4f, 5f}, 1, 2).withGrad(true);
        Tensor sum = a.add(b);
        Tensor prod = sum.mul(a);
        Tensor loss = prod.mean();
        loss.backward();
        assertGradPresent(a);
        assertGradPresent(b);
    }
}
