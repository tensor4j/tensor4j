/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.support;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.Arrays;
import com.github.tensor4j.core.Shape;
import com.github.tensor4j.core.Tensor;

/** JUnit helpers for tensor math validation (tinygrad-style assert_close). */
public final class TensorAssert {

    public static final float DEFAULT_EPS = 1e-5f;

    private TensorAssert() {
    }

    public static void assertShape(int[] expected, Tensor tensor) {
        assertEquals(Arrays.toString(expected), tensor.shape().toString());
    }

    public static void assertAllClose(float[] expected, Tensor actual, float eps) {
        assertEquals(expected.length, actual.numel(), "element count");
        for (int i = 0; i < expected.length; i++) {
            assertClose(expected[i], actual.data()[i], eps, "index " + i);
        }
    }

    public static void assertAllClose(float[] expected, Tensor actual) {
        assertAllClose(expected, actual, DEFAULT_EPS);
    }

    public static void assertAllClose(float[] expected, float[] actual, float eps) {
        assertEquals(expected.length, actual.length, "element count");
        for (int i = 0; i < expected.length; i++) {
            assertClose(expected[i], actual[i], eps, "index " + i);
        }
    }

    public static void assertAllClose(float[] expected, float[] actual) {
        assertAllClose(expected, actual, DEFAULT_EPS);
    }

    public static void assertAllClose(Tensor expected, Tensor actual, float eps) {
        assertShape(expected.shape().dims(), actual);
        assertAllClose(expected.data(), actual, eps);
    }

    public static void assertClose(float expected, float actual, float eps) {
        assertClose(expected, actual, eps, "value");
    }

    public static void assertClose(float expected, float actual, float eps, String message) {
        if (Float.isNaN(expected) && Float.isNaN(actual)) {
            return;
        }
        float delta = Math.abs(expected - actual);
        if (delta > eps) {
            fail(message + ": expected " + expected + " but was " + actual + " (delta " + delta + ")");
        }
    }

    public static void assertNoNaN(Tensor tensor) {
        for (float value : tensor.data()) {
            assertFalse(Float.isNaN(value), "unexpected NaN in tensor");
        }
    }

    public static Tensor matrix2d(float[][] values) {
        int rows = values.length;
        int cols = values[0].length;
        float[] flat = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                flat[r * cols + c] = values[r][c];
            }
        }
        return Tensor.of(flat, rows, cols);
    }

    public static float[] flatten2d(float[][] values) {
        int rows = values.length;
        int cols = values[0].length;
        float[] flat = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(values[r], 0, flat, r * cols, cols);
        }
        return flat;
    }

    public static void assertGradPresent(Tensor tensor) {
        assertTrue(tensor.requiresGrad(), "requiresGrad");
        assertTrue(tensor.grad() != null, "grad should be populated after backward");
        assertNoNaN(tensor.grad());
    }
}
