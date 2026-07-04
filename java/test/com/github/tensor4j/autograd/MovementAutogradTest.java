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
import static org.junit.jupiter.api.Assertions.assertNotNull;

import com.github.tensor4j.core.Tensor;
import org.junit.jupiter.api.Test;

/** Autograd for movement / reduction ops. */
class MovementAutogradTest {

    @Test
    void sumBackwardBroadcastsGradient() {
        Tensor x = Tensor.of(new float[] {1f, 2f, 3f}, 3).withGrad(true);
        Tensor loss = x.sum();
        loss.backward();
        assertGradPresent(x);
        assertAllClose(new float[] {1f, 1f, 1f}, x.grad());
    }

    @Test
    void expandBackwardSumsBroadcastAxes() {
        Tensor x = Tensor.of(new float[] {2f, 3f}, 1, 2).withGrad(true);
        Tensor y = x.expand(3, 1, 2);
        Tensor loss = y.sum();
        loss.backward();
        assertNotNull(x.grad());
        assertAllClose(new float[] {3f, 3f}, x.grad());
    }

    @Test
    void permuteBackwardInvertsAxisOrder() {
        Tensor x = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2).withGrad(true);
        Tensor y = x.permute(1, 0);
        Tensor loss = y.sum();
        loss.backward();
        assertGradPresent(x);
        assertAllClose(new float[] {1f, 1f, 1f, 1f}, x.grad());
    }

    @Test
    void reshapeBackwardRestoresInputLayout() {
        Tensor x = Tensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2).withGrad(true);
        Tensor y = x.reshape(4);
        Tensor loss = y.sum();
        loss.backward();
        assertAllClose(new float[] {1f, 1f, 1f, 1f}, x.grad());
    }
}
