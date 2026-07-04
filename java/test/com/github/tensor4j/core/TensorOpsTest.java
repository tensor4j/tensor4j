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
import static com.github.tensor4j.support.TensorAssert.assertShape;
import static com.github.tensor4j.support.TensorAssert.matrix2d;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Layer 1: op behavior (tinygrad-inspired expected outputs, Java verification). */
class TensorOpsTest {

    @Test
    void matmulSimple2x2() {
        Tensor a = matrix2d(new float[][] {{1f, 2f}, {3f, 4f}});
        Tensor b = matrix2d(new float[][] {{5f, 6f}, {7f, 8f}});
        Tensor c = a.matmul(b);
        assertShape(new int[] {2, 2}, c);
        assertAllClose(new float[] {19f, 22f, 43f, 50f}, c);
    }

    @Test
    void reluZerosNegativeValues() {
        Tensor input = Tensor.of(new float[] {-2f, -1f, 0f, 1f, 2f}, 5);
        Tensor output = input.relu();
        assertAllClose(new float[] {0f, 0f, 0f, 1f, 2f}, output);
    }

    @Test
    void addAndMulElementwise() {
        Tensor left = Tensor.of(new float[] {1f, 2f, 3f}, 3);
        Tensor right = Tensor.of(new float[] {4f, 5f, 6f}, 3);
        assertAllClose(new float[] {5f, 7f, 9f}, left.add(right));
        assertAllClose(new float[] {4f, 10f, 18f}, left.mul(right));
    }

    @Test
    void matmulRejectsInnerDimMismatch() {
        Tensor a = matrix2d(new float[][] {{1f, 2f, 3f}});
        Tensor b = matrix2d(new float[][] {{1f, 2f}, {3f, 4f}});
        assertThrows(IllegalArgumentException.class, new MatmulMismatchAction(a, b));
    }

    @Test
    void reshapeIsZeroCopyViewWhenContiguous() {
        Tensor source = Tensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 2, 3);
        Tensor reshaped = source.reshape(3, 2);
        assertShape(new int[] {3, 2}, reshaped);
        assertAllClose(source.data(), reshaped);
        assertTrue(source.buffer().data() == reshaped.buffer().data(), "reshape should share root buffer");
        reshaped.set(99f, 0, 0);
        assertAllClose(new float[] {99f, 2f, 3f, 4f, 5f, 6f}, source);
    }

    @Test
    void transpose2dIsZeroCopyView() {
        Tensor matrix = matrix2d(new float[][] {{1f, 2f, 3f}, {4f, 5f, 6f}});
        Tensor transposed = matrix.transpose2d();
        assertShape(new int[] {3, 2}, transposed);
        assertTrue(matrix.buffer().data() == transposed.buffer().data(), "transpose should share root buffer");
        assertAllClose(new float[] {1f, 4f, 2f, 5f, 3f, 6f}, transposed);
        transposed.set(0f, 1, 0);
        assertClose(0f, matrix.get(0, 1), 1e-5f);
    }

    @Test
    void stridesMatchRowMajorLayout() {
        Tensor matrix = matrix2d(new float[][] {{1f, 2f}, {3f, 4f}});
        assertTrue(matrix.isContiguous());
        org.junit.jupiter.api.Assertions.assertArrayEquals(new int[] {2, 1}, matrix.strides());
    }

    private static void assertClose(float expected, float actual, float eps) {
        com.github.tensor4j.support.TensorAssert.assertClose(expected, actual, eps);
    }

    private static final class MatmulMismatchAction implements org.junit.jupiter.api.function.Executable {
        private final Tensor left;
        private final Tensor right;

        MatmulMismatchAction(Tensor left, Tensor right) {
            this.left = left;
            this.right = right;
        }

        @Override
        public void execute() {
            left.matmul(right);
        }
    }
}
