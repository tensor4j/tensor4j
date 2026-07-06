/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.gpu;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import com.github.tensor4j.gpu.core.*;
import com.github.tensor4j.gpu.ref.*;

/**
 * GPU math operation tests using CPU reference backend.
 * Covers element-wise and reduction ops. CI-safe, no GPU needed.
 */
class GpuTensorMathTest {

    @Test
    void addTwoVectors() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f, 5f}, 5);
            GpuTensor b = GpuTensor.fromHost(dev, new float[]{5f, 4f, 3f, 2f, 1f}, 5);
            GpuTensor c = math.add(a, b);
            assertArrayEquals(new float[]{6f, 6f, 6f, 6f, 6f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void addEmpty() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{}, 0);
            GpuTensor b = GpuTensor.fromHost(dev, new float[]{}, 0);
            GpuTensor c = math.add(a, b);
            assertEquals(0, c.numel());
        }
    }

    @Test
    void addRejectsShapeMismatch() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f}, 2);
            GpuTensor b = GpuTensor.fromHost(dev, new float[]{3f, 4f, 5f}, 3);
            assertThrows(IllegalArgumentException.class, () -> math.add(a, b));
        }
    }

    @Test
    void subTwoVectors() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{10f, 20f, 30f}, 3);
            GpuTensor b = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 3);
            GpuTensor c = math.sub(a, b);
            assertArrayEquals(new float[]{9f, 18f, 27f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void subRejectsShapeMismatch() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            assertThrows(IllegalArgumentException.class,
                () -> math.sub(
                    GpuTensor.fromHost(dev, new float[]{1f}, 1),
                    GpuTensor.fromHost(dev, new float[]{1f, 2f}, 2)));
        }
    }

    @Test
    void mulTwoVectors() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f}, 4);
            GpuTensor b = GpuTensor.fromHost(dev, new float[]{2f, 3f, 4f, 5f}, 4);
            GpuTensor c = math.mul(a, b);
            assertArrayEquals(new float[]{2f, 6f, 12f, 20f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void divTwoVectors() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{10f, 20f, 30f}, 3);
            GpuTensor b = GpuTensor.fromHost(dev, new float[]{2f, 4f, 5f}, 3);
            GpuTensor c = math.div(a, b);
            assertArrayEquals(new float[]{5f, 5f, 6f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void divByZeroGivesInfinity() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f}, 2);
            GpuTensor b = GpuTensor.fromHost(dev, new float[]{0f, 0f}, 2);
            GpuTensor c = math.div(a, b);
            float[] result = c.toHost();
            assertTrue(Float.isInfinite(result[0]));
            assertTrue(Float.isInfinite(result[1]));
        }
    }

    @Test
    void reluAllPositive() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f, 5f}, 5);
            GpuTensor c = math.relu(a);
            assertArrayEquals(new float[]{1f, 2f, 3f, 4f, 5f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void reluMixed() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{-2f, -1f, 0f, 1f, 2f}, 5);
            GpuTensor c = math.relu(a);
            assertArrayEquals(new float[]{0f, 0f, 0f, 1f, 2f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void reluAllNegative() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{-5f, -4f, -3f}, 3);
            GpuTensor c = math.relu(a);
            assertArrayEquals(new float[]{0f, 0f, 0f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void neg() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, -2f, 0f, -3f, 5f}, 5);
            GpuTensor c = math.neg(a);
            assertArrayEquals(new float[]{-1f, 2f, 0f, 3f, -5f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void scale() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f}, 4);
            GpuTensor c = math.scale(a, 2.5f);
            assertArrayEquals(new float[]{2.5f, 5f, 7.5f, 10f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void scaleByZero() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 3);
            GpuTensor c = math.scale(a, 0f);
            assertArrayEquals(new float[]{0f, 0f, 0f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void scaleByOne() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{4f, 5f, 6f}, 3);
            GpuTensor c = math.scale(a, 1f);
            assertArrayEquals(new float[]{4f, 5f, 6f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void exp() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{0f, 1f, 2f}, 3);
            GpuTensor c = math.exp(a);
            float[] expected = {(float) Math.exp(0), (float) Math.exp(1), (float) Math.exp(2)};
            assertArrayEquals(expected, c.toHost(), 1e-5f);
        }
    }

    @Test
    void expNegative() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{-1f, -2f, -3f}, 3);
            GpuTensor c = math.exp(a);
            float[] expected = {(float) Math.exp(-1), (float) Math.exp(-2), (float) Math.exp(-3)};
            assertArrayEquals(expected, c.toHost(), 1e-5f);
        }
    }

    @Test
    void log() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, (float) Math.E}, 3);
            GpuTensor c = math.log(a);
            float[] expected = {(float) Math.log(1), (float) Math.log(2), (float) Math.log(Math.E)};
            assertArrayEquals(expected, c.toHost(), 1e-5f);
        }
    }

    @Test
    void logOfOne() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f}, 1);
            GpuTensor c = math.log(a);
            assertArrayEquals(new float[]{0f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void matmul2x2() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f}, 2, 2);
            GpuTensor b = GpuTensor.fromHost(dev, new float[]{5f, 6f, 7f, 8f}, 2, 2);
            GpuTensor c = math.matmul(a, b);
            assertArrayEquals(new int[]{2, 2}, c.shape());
            assertArrayEquals(new float[]{19f, 22f, 43f, 50f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void matmulIdentity() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3);
            GpuTensor eye = GpuTensor.fromHost(dev, new float[]{1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f}, 3, 3);
            GpuTensor c = math.matmul(a, eye);
            assertArrayEquals(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void matmulNonSquare() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f}, 2, 2);
            GpuTensor b = GpuTensor.fromHost(dev, new float[]{1f, 0f}, 2, 1);
            GpuTensor c = math.matmul(a, b);
            assertArrayEquals(new int[]{2, 1}, c.shape());
            assertArrayEquals(new float[]{1f, 3f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void matmulRejectsNon2D() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 3);
            GpuTensor b = GpuTensor.fromHost(dev, new float[]{4f, 5f, 6f}, 3);
            assertThrows(IllegalArgumentException.class, () -> math.matmul(a, b));
        }
    }

    @Test
    void matmulRejectsInnerDimMismatch() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f}, 2, 2);
            GpuTensor b = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 3, 2);
            assertThrows(IllegalArgumentException.class, () -> math.matmul(a, b));
        }
    }

    @Test
    void matmulLarge() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            int m = 8;
            int k = 16;
            int n = 4;
            float[] dataA = new float[m * k];
            float[] dataB = new float[k * n];
            for (int i = 0; i < dataA.length; i++) dataA[i] = 1f;
            for (int i = 0; i < dataB.length; i++) dataB[i] = 2f;
            GpuTensor a = GpuTensor.fromHost(dev, dataA, m, k);
            GpuTensor b = GpuTensor.fromHost(dev, dataB, k, n);
            GpuTensor c = math.matmul(a, b);
            float[] expected = new float[m * n];
            for (int i = 0; i < m * n; i++) expected[i] = 2f * k;
            assertArrayEquals(expected, c.toHost(), 1e-5f);
        }
    }

    @Test
    void chainedOpsGivesCorrectResult() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f}, 4);
            GpuTensor b = GpuTensor.fromHost(dev, new float[]{5f, 6f, 7f, 8f}, 4);
            GpuTensor step1 = math.add(a, b);
            GpuTensor step2 = math.scale(step1, 0.5f);
            GpuTensor step3 = math.relu(step2);
            assertArrayEquals(new float[]{3f, 4f, 5f, 6f}, step3.toHost(), 1e-5f);
        }
    }

    @Test
    void chainAddSubMulDiv() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{10f}, 1);
            GpuTensor b = GpuTensor.fromHost(dev, new float[]{2f}, 1);
            GpuTensor c = GpuTensor.fromHost(dev, new float[]{3f}, 1);
            GpuTensor t = math.div(math.mul(math.sub(a, b), c), b);
            assertArrayEquals(new float[]{12f}, t.toHost(), 1e-5f);
        }
    }

    @Test
    void mathReturnsCorrectDevice() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            assertSame(dev, math.device());
        }
    }

    @Test
    void multipleAddCalls() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            float[] dataA = {10f, 20f, 30f};
            float[] dataB = {1f, 2f, 3f};
            GpuTensor a = GpuTensor.fromHost(dev, dataA, 3);
            GpuTensor b = GpuTensor.fromHost(dev, dataB, 3);
            for (int i = 0; i < 5; i++) {
                GpuTensor c = math.add(a, b);
                assertArrayEquals(new float[]{11f, 22f, 33f}, c.toHost(), 1e-5f);
            }
        }
    }

    @Test
    void sqrt() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{0f, 1f, 4f, 9f, 16f}, 5);
            GpuTensor c = math.sqrt(a);
            assertArrayEquals(new float[]{0f, 1f, 2f, 3f, 4f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void sigmoid() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{-2f, -1f, 0f, 1f, 2f}, 5);
            GpuTensor c = math.sigmoid(a);
            float[] expected = {
                1f / (1f + (float) Math.exp(2)),
                1f / (1f + (float) Math.exp(1)),
                0.5f,
                1f / (1f + (float) Math.exp(-1)),
                1f / (1f + (float) Math.exp(-2))
            };
            assertArrayEquals(expected, c.toHost(), 1e-5f);
        }
    }

    @Test
    void tanh() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{-2f, -1f, 0f, 1f, 2f}, 5);
            GpuTensor c = math.tanh(a);
            float[] expected = {
                (float) Math.tanh(-2), (float) Math.tanh(-1), 0f,
                (float) Math.tanh(1), (float) Math.tanh(2)
            };
            assertArrayEquals(expected, c.toHost(), 1e-5f);
        }
    }

    @Test
    void squaredDifference() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{5f, 5f, 5f}, 3);
            GpuTensor b = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 3);
            GpuTensor c = math.squaredDifference(a, b);
            assertArrayEquals(new float[]{16f, 9f, 4f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void sum() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f, 5f}, 5);
            GpuTensor c = math.sum(a);
            assertArrayEquals(new float[]{15f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void sumLarge() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            float[] data = new float[1000];
            float expected = 0f;
            for (int i = 0; i < 1000; i++) {
                data[i] = i * 0.5f;
                expected += i * 0.5f;
            }
            GpuTensor a = GpuTensor.fromHost(dev, data, 1000);
            GpuTensor c = math.sum(a);
            assertArrayEquals(new float[]{expected}, c.toHost(), 1e-3f);
        }
    }

    @Test
    void softmax() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 3);
            GpuTensor c = math.softmax(a);
            float[] hc = c.toHost();
            assertEquals(3, hc.length);
            float sum = 0f;
            for (float v : hc) {
                sum += v;
            }
            assertEquals(1f, sum, 1e-5f);
            assertTrue(hc[2] > hc[1]);
            assertTrue(hc[1] > hc[0]);
        }
    }

    @Test
    void softmaxNegativeValues() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{-1f, -2f, -3f}, 3);
            GpuTensor c = math.softmax(a);
            float[] hc = c.toHost();
            float sum = 0f;
            for (float v : hc) {
                sum += v;
            }
            assertEquals(1f, sum, 1e-5f);
            assertTrue(hc[0] > hc[1]);
            assertTrue(hc[1] > hc[2]);
        }
    }

    @Test
    void powElementWise() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 3);
            GpuTensor b = GpuTensor.fromHost(dev, new float[]{2f, 3f, 2f}, 3);
            GpuTensor c = math.pow(a, b);
            assertArrayEquals(new float[]{1f, 8f, 9f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void meanVector() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{2f, 4f, 6f}, 3);
            GpuTensor c = math.mean(a);
            assertEquals(4f, c.toHost()[0], 1e-5f);
        }
    }

    @Test
    void absVector() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{-3f, -1f, 0f, 2f, 5f}, 5);
            GpuTensor c = math.abs(a);
            assertArrayEquals(new float[]{3f, 1f, 0f, 2f, 5f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void clipValues() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{-5f, -1f, 0f, 3f, 10f}, 5);
            GpuTensor c = math.clip(a, -2f, 4f);
            assertArrayEquals(new float[]{-2f, -1f, 0f, 3f, 4f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void transposeMatrix() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f}, 2, 2);
            GpuTensor c = math.transpose(a);
            assertArrayEquals(new float[]{1f, 3f, 2f, 4f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void transposeNonSquare() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3);
            GpuTensor c = math.transpose(a);
            assertArrayEquals(new int[]{3, 2}, c.shape());
            assertArrayEquals(new float[]{1f, 4f, 2f, 5f, 3f, 6f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void transposeRejectsNon2D() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 3);
            assertThrows(IllegalArgumentException.class, () -> math.transpose(a));
        }
    }

    @Test
    void sinVector() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{0f, (float) (Math.PI / 2), (float) Math.PI}, 3);
            GpuTensor c = math.sin(a);
            assertArrayEquals(new float[]{0f, 1f, 0f}, c.toHost(), 1e-4f);
        }
    }

    @Test
    void cosVector() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{0f, (float) (Math.PI / 2), (float) Math.PI}, 3);
            GpuTensor c = math.cos(a);
            assertArrayEquals(new float[]{1f, 0f, -1f}, c.toHost(), 1e-4f);
        }
    }

    @Test
    void geluForward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{-2f, -1f, 0f, 1f, 2f}, 5);
            GpuTensor c = math.gelu(a);
            float[] h = c.toHost();
            // GELU is roughly identity near 0 for positive, ~0 for very negative
            assertEquals(0f, h[0], 0.1f); // gelu(-2) ≈ -0.045
            assertTrue(h[3] > 0.8f && h[3] < 1.2f); // gelu(1) ≈ 0.84
            assertTrue(h[4] > 1.8f && h[4] < 2.2f); // gelu(2) ≈ 1.95
        }
    }

    @Test
    void leakyReluForward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{-2f, -1f, 0f, 1f, 2f}, 5);
            GpuTensor c = math.leakyRelu(a, 0.1f);
            assertArrayEquals(new float[]{-0.2f, -0.1f, 0f, 1f, 2f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void expandBroadcastsSingleton() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 1, 3);
            GpuTensor c = math.expand(a, 2, 3);
            assertArrayEquals(new int[]{2, 3}, c.shape());
            assertArrayEquals(new float[]{1f, 2f, 3f, 1f, 2f, 3f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void expandScalarToVector() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{5f}, 1);
            GpuTensor c = math.expand(a, 4);
            assertArrayEquals(new float[]{5f, 5f, 5f, 5f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void dropoutPreservesSumInExpectation() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{10f, 20f, 30f, 40f}, 4);
            java.util.Random rng = new java.util.Random(42);
            float prob = 0.5f;
            float sum = 0f;
            int trials = 100;
            for (int t = 0; t < trials; t++) {
                GpuTensor c = math.dropout(a, prob, new java.util.Random(42 + t));
                for (float v : c.toHost()) sum += v;
            }
            float avgSum = sum / trials;
            float expectedSum = 100f; // (10+20+30+40) = 100
            assertEquals(expectedSum, avgSum, expectedSum * 0.2f);
        }
    }

    @Test
    void dropoutZeroProbIsIdentity() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 3);
            GpuTensor c = math.dropout(a, 0f, new java.util.Random(42));
            assertArrayEquals(new float[]{1f, 2f, 3f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void binaryCrossEntropyCorrectValues() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor pred = GpuTensor.fromHost(dev, new float[]{0.9f, 0.2f, 0.8f}, 3);
            GpuTensor target = GpuTensor.fromHost(dev, new float[]{1f, 0f, 1f}, 3);
            GpuTensor loss = math.binaryCrossEntropy(pred, target);
            float l = loss.toHost()[0];
            // Manual: -ln(0.9) - ln(0.8) - ln(0.8) = -(ln0.9 + 2*ln0.8)
            float expected = -(float)(Math.log(0.9) + 2 * Math.log(0.8));
            assertEquals(expected, l, 1e-5f);
        }
    }

    @Test
    void binaryCrossEntropyPerfectPrediction() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor pred = GpuTensor.fromHost(dev, new float[]{0.9999999f, 0.0000001f}, 2);
            GpuTensor target = GpuTensor.fromHost(dev, new float[]{1f, 0f}, 2);
            GpuTensor loss = math.binaryCrossEntropy(pred, target);
            float l = loss.toHost()[0];
            assertEquals(0f, l, 1e-3f);
        }
    }

    @Test
    void softmax2dRowsSumToOne() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3);
            GpuTensor c = math.softmax(a);
            assertArrayEquals(new int[]{2, 3}, c.shape());
            float[] hc = c.toHost();
            for (int row = 0; row < 2; row++) {
                float sum = 0f;
                for (int col = 0; col < 3; col++) {
                    sum += hc[row * 3 + col];
                }
                assertEquals(1f, sum, 1e-5f);
            }
        }
    }

    @Test
    void crossEntropyCorrectValue() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor logits = GpuTensor.fromHost(dev, new float[]{2f, 1f, 0.1f, 0f, 3f, 1f}, 2, 3);
            GpuTensor targets = GpuTensor.fromHost(dev, new float[]{0f, 1f}, 2);
            GpuTensor loss = math.crossEntropy(logits, targets);
            float l = loss.toHost()[0];
            // Manual computation: softmax row 0 = [0.659, 0.242, 0.099], target=0
            // -ln(0.659) = 0.417
            // softmax row 1 = [0.042, 0.844, 0.114], target=1
            // -ln(0.844) = 0.170
            // sum = 0.587
            assertTrue(l > 0.5f && l < 0.7f, "crossEntropy should be ~0.587, got " + l);
        }
    }

    @Test
    void layerNormZeroMeanUnitVar() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor x = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3);
            GpuTensor gamma = GpuTensor.fromHost(dev, new float[]{1f, 1f, 1f}, 3);
            GpuTensor beta = GpuTensor.fromHost(dev, new float[]{0f, 0f, 0f}, 3);
            GpuTensor c = math.layerNorm(x, gamma, beta, 1e-5f);
            float[] hc = c.toHost();
            for (int row = 0; row < 2; row++) {
                float mean = 0f;
                for (int col = 0; col < 3; col++) mean += hc[row * 3 + col];
                mean /= 3;
                assertEquals(0f, mean, 0.1f);
            }
        }
    }

    @Test
    void siluForward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{-1f, 0f, 1f, 2f}, 4);
            GpuTensor c = math.silu(a);
            float[] h = c.toHost();
            // silu(0) = 0, silu(x) = x*sigmoid(x)
            assertEquals(0f, h[1], 1e-5f);
            assertTrue(h[2] > 0.7f && h[2] < 0.8f); // silu(1) ≈ 0.731
        }
    }

    @Test
    void eluForward() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{-1f, 0f, 1f, 2f}, 4);
            GpuTensor c = math.elu(a, 1f);
            float[] h = c.toHost();
            assertEquals(0f, h[1], 1e-5f); // elu(0) = 0
            assertEquals(1f, h[2], 1e-5f); // elu(1) = 1
            assertTrue(h[0] > -0.64f && h[0] < -0.62f); // elu(-1) ≈ -0.632
        }
    }

    @Test
    void flatten2dTo1d() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3);
            GpuTensor c = math.flatten(a, 0);
            assertArrayEquals(new int[]{6}, c.shape());
            assertArrayEquals(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void argmax2d() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor a = GpuTensor.fromHost(dev, new float[]{0.1f, 0.9f, 0.2f, 0.5f, 0.3f, 0.8f}, 2, 3);
            GpuTensor c = math.argmax(a, 1);
            assertArrayEquals(new int[]{2}, c.shape());
            float[] h = c.toHost();
            assertEquals(1f, h[0], 1e-5f); // max of [0.1, 0.9, 0.2] is at index 1
            assertEquals(2f, h[1], 1e-5f); // max of [0.5, 0.3, 0.8] is at index 2
        }
    }

    @Test
    void oneHotEncoding() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor indices = GpuTensor.fromHost(dev, new float[]{0f, 2f, 1f}, 3);
            GpuTensor c = math.oneHot(indices, 3);
            assertArrayEquals(new int[]{3, 3}, c.shape());
            assertArrayEquals(new float[]{1f, 0f, 0f, 0f, 0f, 1f, 0f, 1f, 0f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void mseLossCorrectValue() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor pred = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 3);
            GpuTensor target = GpuTensor.fromHost(dev, new float[]{1f, 0f, 3f}, 3);
            GpuTensor loss = math.mseLoss(pred, target);
            float l = loss.toHost()[0];
            // (1-1)^2 + (2-0)^2 + (3-3)^2 = 0 + 4 + 0 = 4
            assertEquals(4f, l, 1e-5f);
        }
    }

    @Test
    void embeddingLookup() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor weight = GpuTensor.fromHost(dev, new float[]{10f, 11f, 20f, 21f, 30f, 31f}, 3, 2);
            GpuTensor indices = GpuTensor.fromHost(dev, new float[]{0f, 2f, 1f}, 3);
            GpuTensor c = math.embedding(weight, indices);
            assertArrayEquals(new int[]{3, 2}, c.shape());
            assertArrayEquals(new float[]{10f, 11f, 30f, 31f, 20f, 21f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void conv2dSingleImage() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            // 1 image, 1 channel, 3x3 input
            GpuTensor input = GpuTensor.fromHost(dev, new float[]{1,2,3,4,5,6,7,8,9}, 1, 1, 3, 3);
            // 1 output channel, 1 input channel, 2x2 kernel
            GpuTensor kernel = GpuTensor.fromHost(dev, new float[]{1,0,0,1}, 1, 1, 2, 2);
            GpuTensor c = math.conv2d(input, kernel, 1, 0);
            assertArrayEquals(new int[]{1, 1, 2, 2}, c.shape());
            // [1*1+2*0+4*0+5*1, 2*1+3*0+5*0+6*1, 4*1+5*0+7*0+8*1, 5*1+6*0+8*0+9*1]
            assertArrayEquals(new float[]{6f, 8f, 12f, 14f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void maxPool2dSimple() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor input = GpuTensor.fromHost(dev, new float[]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, 1, 1, 4, 4);
            GpuTensor c = math.maxPool2d(input, 2, 2, 0);
            assertArrayEquals(new int[]{1, 1, 2, 2}, c.shape());
            // max of [1,2,5,6]=6, max of [3,4,7,8]=8, max of [9,10,13,14]=14, max of [11,12,15,16]=16
            assertArrayEquals(new float[]{6f, 8f, 14f, 16f}, c.toHost(), 1e-5f);
        }
    }

    @Test
    void batchNorm1dPerFeatureNormalized() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensorMath math = new GpuTensorMath(dev);
            GpuTensor x = GpuTensor.fromHost(dev, new float[]{1f, 10f, 2f, 20f, 3f, 30f}, 3, 2);
            GpuTensor gamma = GpuTensor.fromHost(dev, new float[]{1f, 1f}, 2);
            GpuTensor beta = GpuTensor.fromHost(dev, new float[]{0f, 0f}, 2);
            GpuTensor c = math.batchNorm1d(x, gamma, beta, 1e-5f);
            float[] hc = c.toHost();
            // Each column should have ~0 mean
            for (int col = 0; col < 2; col++) {
                float mean = 0f;
                for (int row = 0; row < 3; row++) mean += hc[row * 2 + col];
                mean /= 3;
                assertEquals(0f, mean, 0.1f);
            }
        }
    }
}
