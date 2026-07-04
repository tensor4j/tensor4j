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
import static com.github.tensor4j.support.TensorAssert.assertClose;
import static com.github.tensor4j.support.TensorAssert.assertShape;
import static com.github.tensor4j.support.TensorAssert.matrix2d;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * Layer 1: forward tests for all tensor4j ops (tinygrad op surface at sample scale).
 * Expected values use standard Java {@code float} — same on every platform.
 */
class AllOpsForwardTest {

    private static final float EPS = 1e-5f;

    @Test
    void opZeros() {
        Tensor tensor = Tensor.zeros(2, 3);
        assertShape(new int[] {2, 3}, tensor);
        assertAllClose(new float[] {0f, 0f, 0f, 0f, 0f, 0f}, tensor, EPS);
    }

    @Test
    void opAdd() {
        Tensor left = Tensor.of(new float[] {1f, 2f, 3f}, 3);
        Tensor right = Tensor.of(new float[] {4f, 5f, 6f}, 3);
        assertAllClose(new float[] {5f, 7f, 9f}, left.add(right), EPS);
    }

    @Test
    void opSub() {
        Tensor left = Tensor.of(new float[] {5f, 7f, 9f}, 3);
        Tensor right = Tensor.of(new float[] {1f, 2f, 3f}, 3);
        assertAllClose(new float[] {4f, 5f, 6f}, left.sub(right), EPS);
    }

    @Test
    void opMul() {
        Tensor left = Tensor.of(new float[] {1f, 2f, 3f}, 3);
        Tensor right = Tensor.of(new float[] {4f, 5f, 6f}, 3);
        assertAllClose(new float[] {4f, 10f, 18f}, left.mul(right), EPS);
    }

    @Test
    void opDiv() {
        Tensor left = Tensor.of(new float[] {6f, 8f, 9f}, 3);
        Tensor right = Tensor.of(new float[] {2f, 4f, 3f}, 3);
        assertAllClose(new float[] {3f, 2f, 3f}, left.div(right), EPS);
    }

    @Test
    void opNeg() {
        Tensor input = Tensor.of(new float[] {1f, -2f, 0f}, 3);
        assertAllClose(new float[] {-1f, 2f, 0f}, input.neg(), EPS);
    }

    @Test
    void opMatmul() {
        Tensor a = matrix2d(new float[][] {{1f, 2f}, {3f, 4f}});
        Tensor b = matrix2d(new float[][] {{5f, 6f}, {7f, 8f}});
        assertAllClose(new float[] {19f, 22f, 43f, 50f}, a.matmul(b), EPS);
    }

    @Test
    void opMatmulBatchRows() {
        Tensor batch = matrix2d(new float[][] {{1f, 0f}, {0f, 1f}});
        Tensor weights = matrix2d(new float[][] {{2f, 3f}, {4f, 5f}});
        assertAllClose(new float[] {2f, 3f, 4f, 5f}, batch.matmul(weights), EPS);
    }

    @Test
    void opRelu() {
        Tensor input = Tensor.of(new float[] {-2f, -1f, 0f, 1f, 2f}, 5);
        assertAllClose(new float[] {0f, 0f, 0f, 1f, 2f}, input.relu(), EPS);
    }

    @Test
    void opSum() {
        assertClose(10f, Tensor.of(new float[] {1f, 2f, 3f, 4f}, 4).sum().data()[0], EPS);
    }

    @Test
    void opMean() {
        assertClose(2.5f, Tensor.of(new float[] {1f, 2f, 3f, 4f}, 4).mean().data()[0], EPS);
    }

    @Test
    void opSumAxis() {
        Tensor batch = matrix2d(new float[][] {{1f, 2f, 3f}, {4f, 5f, 6f}});
        assertAllClose(new float[] {5f, 7f, 9f}, batch.sumAxis(0), EPS);
    }

    @Test
    void opReshape() {
        Tensor source = Tensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 2, 3);
        Tensor reshaped = source.reshape(3, 2);
        assertShape(new int[] {3, 2}, reshaped);
        assertAllClose(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, reshaped, EPS);
    }

    @Test
    void opFlatten() {
        Tensor matrix = matrix2d(new float[][] {{1f, 2f}, {3f, 4f}});
        assertAllClose(new float[] {1f, 2f, 3f, 4f}, matrix.flatten(), EPS);
    }

    @Test
    void opExpand() {
        Tensor vector = Tensor.of(new float[] {1f, 2f, 3f}, 3);
        assertAllClose(new float[] {1f, 2f, 3f, 1f, 2f, 3f}, vector.expand(2, 3), EPS);
    }

    @Test
    void opPermute() {
        Tensor cube = Tensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, 2, 2, 2);
        assertAllClose(new float[] {1f, 3f, 5f, 7f, 2f, 4f, 6f, 8f}, cube.permute(2, 0, 1), EPS);
    }

    @Test
    void opTranspose() {
        Tensor matrix = matrix2d(new float[][] {{1f, 2f, 3f}, {4f, 5f, 6f}});
        assertAllClose(new float[] {1f, 4f, 2f, 5f, 3f, 6f}, matrix.transpose2d(), EPS);
    }

    @Test
    void opContiguous() {
        Tensor view = matrix2d(new float[][] {{1f, 2f, 3f}, {4f, 5f, 6f}}).transpose2d();
        Tensor dense = view.contiguous();
        assertTrue(dense.isContiguous());
        assertAllClose(new float[] {1f, 4f, 2f, 5f, 3f, 6f}, dense, EPS);
    }

    @Test
    void opDetach() {
        Tensor source = Tensor.of(new float[] {1f, 2f}, 2).withGrad(true);
        Tensor copy = source.detach();
        assertFalse(copy.requiresGrad());
        assertAllClose(new float[] {1f, 2f}, copy, EPS);
    }

    @Test
    void opPow2() {
        Tensor input = Tensor.of(new float[] {1f, 2f, 3f}, 3);
        assertAllClose(new float[] {1f, 4f, 9f}, input.pow2(), EPS);
    }
}
