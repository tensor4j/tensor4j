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
import com.github.tensor4j.core.Tensor;
import com.github.tensor4j.gpu.core.GpuBridge;
import com.github.tensor4j.gpu.ref.CpuDevice;

/**
 * Tests for CPU-Tensor / GPU-GpuTensor bridge.
 * Uses CPU reference device so tests run in CI.
 */
class GpuBridgeTest {

    @Test
    void bridgeAdd() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBridge bridge = new GpuBridge(dev);
            Tensor a = Tensor.of(new float[]{1f, 2f, 3f}, 3);
            Tensor b = Tensor.of(new float[]{4f, 5f, 6f}, 3);
            Tensor c = bridge.add(a, b);
            assertArrayEquals(new float[]{5f, 7f, 9f}, c.data(), 1e-5f);
        }
    }

    @Test
    void bridgeSub() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBridge bridge = new GpuBridge(dev);
            Tensor a = Tensor.of(new float[]{10f, 20f, 30f}, 3);
            Tensor b = Tensor.of(new float[]{1f, 2f, 3f}, 3);
            Tensor c = bridge.sub(a, b);
            assertArrayEquals(new float[]{9f, 18f, 27f}, c.data(), 1e-5f);
        }
    }

    @Test
    void bridgeMul() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBridge bridge = new GpuBridge(dev);
            Tensor a = Tensor.of(new float[]{1f, 2f, 3f, 4f}, 4);
            Tensor b = Tensor.of(new float[]{2f, 3f, 4f, 5f}, 4);
            Tensor c = bridge.mul(a, b);
            assertArrayEquals(new float[]{2f, 6f, 12f, 20f}, c.data(), 1e-5f);
        }
    }

    @Test
    void bridgeDiv() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBridge bridge = new GpuBridge(dev);
            Tensor a = Tensor.of(new float[]{10f, 20f, 30f}, 3);
            Tensor b = Tensor.of(new float[]{2f, 4f, 5f}, 3);
            Tensor c = bridge.div(a, b);
            assertArrayEquals(new float[]{5f, 5f, 6f}, c.data(), 1e-5f);
        }
    }

    @Test
    void bridgeRelu() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBridge bridge = new GpuBridge(dev);
            Tensor a = Tensor.of(new float[]{-2f, -1f, 0f, 1f, 2f}, 5);
            Tensor c = bridge.relu(a);
            assertArrayEquals(new float[]{0f, 0f, 0f, 1f, 2f}, c.data(), 1e-5f);
        }
    }

    @Test
    void bridgeNeg() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBridge bridge = new GpuBridge(dev);
            Tensor a = Tensor.of(new float[]{1f, -2f, 0f}, 3);
            Tensor c = bridge.neg(a);
            assertArrayEquals(new float[]{-1f, 2f, 0f}, c.data(), 1e-5f);
        }
    }

    @Test
    void bridgeScale() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBridge bridge = new GpuBridge(dev);
            Tensor a = Tensor.of(new float[]{1f, 2f, 3f}, 3);
            Tensor c = bridge.scale(a, 2.5f);
            assertArrayEquals(new float[]{2.5f, 5f, 7.5f}, c.data(), 1e-5f);
        }
    }

    @Test
    void bridgeExp() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBridge bridge = new GpuBridge(dev);
            Tensor a = Tensor.of(new float[]{0f, 1f, 2f}, 3);
            Tensor c = bridge.exp(a);
            float[] expected = {(float) Math.exp(0), (float) Math.exp(1), (float) Math.exp(2)};
            assertArrayEquals(expected, c.data(), 1e-5f);
        }
    }

    @Test
    void bridgeLog() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBridge bridge = new GpuBridge(dev);
            Tensor a = Tensor.of(new float[]{1f, 2f, (float) Math.E}, 3);
            Tensor c = bridge.log(a);
            float[] expected = {(float) Math.log(1), (float) Math.log(2), (float) Math.log(Math.E)};
            assertArrayEquals(expected, c.data(), 1e-5f);
        }
    }

    @Test
    void bridgeMatmul() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBridge bridge = new GpuBridge(dev);
            Tensor a = Tensor.of(new float[]{1f, 2f, 3f, 4f}, 2, 2);
            Tensor b = Tensor.of(new float[]{5f, 6f, 7f, 8f}, 2, 2);
            Tensor c = bridge.matmul(a, b);
            assertArrayEquals(new float[]{19f, 22f, 43f, 50f}, c.data(), 1e-5f);
        }
    }

    @Test
    void bridgePreservesShape() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBridge bridge = new GpuBridge(dev);
            Tensor a = Tensor.of(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3);
            Tensor b = Tensor.of(new float[]{6f, 5f, 4f, 3f, 2f, 1f}, 2, 3);
            Tensor c = bridge.add(a, b);
            assertArrayEquals(new int[]{2, 3}, c.shape().dims());
        }
    }

    @Test
    void bridgeReturnsCorrectDevice() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBridge bridge = new GpuBridge(dev);
            assertSame(dev, bridge.device());
        }
    }

    @Test
    void bridgeRoundTripToGpuAndBack() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBridge bridge = new GpuBridge(dev);
            Tensor a = Tensor.of(new float[]{1f, 2f, 3f, 4f}, 2, 2);
            try (var ga = bridge.toGpu(a)) {
                assertArrayEquals(new int[]{2, 2}, ga.shape());
                Tensor back = bridge.toTensor(ga);
                assertArrayEquals(a.data(), back.data(), 1e-5f);
            }
        }
    }
}
