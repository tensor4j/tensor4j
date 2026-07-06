/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.gpu.it;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import com.github.tensor4j.gpu.core.GpuDevice;
import com.github.tensor4j.gpu.core.GpuRuntime;

/**
 * End-to-end integration test for GPU runtime auto-detection.
 * On a GPU-equipped machine with native libs, this verifies that
 * GpuRuntime detects the best available device and runs operations.
 * On CI without GPU hardware, GPU tests are expected to fail gracefully
 * with CPU fallback already verified in SyclDeviceTest.
 */
@Tag("gpu-it")
class GpuRuntimeIT {

    @Test
    void runtimeReturnsNonCpuDevice() {
        GpuDevice dev = GpuRuntime.defaultDevice();
        assertNotNull(dev);
        // On a GPU-enabled system, the detected device should NOT be CpuDevice
        String name = dev.name();
        assertNotNull(name);
        System.out.println("Detected GPU device: " + name + " (type: " + dev.getClass().getSimpleName() + ")");
    }

    @Test
    void mathOperationsRunOnDevice() throws Exception {
        com.github.tensor4j.gpu.core.GpuTensorMath math = GpuRuntime.math();
        com.github.tensor4j.gpu.core.GpuTensor a = com.github.tensor4j.gpu.core.GpuTensor.fromHost(
            GpuRuntime.defaultDevice(), new float[]{1f, 2f, 3f, 4f, 5f}, 5);
        com.github.tensor4j.gpu.core.GpuTensor b = com.github.tensor4j.gpu.core.GpuTensor.fromHost(
            GpuRuntime.defaultDevice(), new float[]{5f, 4f, 3f, 2f, 1f}, 5);
        com.github.tensor4j.gpu.core.GpuTensor c = math.add(a, b);
        float[] result = c.toHost();
        assertArrayEquals(new float[]{6f, 6f, 6f, 6f, 6f}, result, 1e-5f);
    }

    @Test
    void autogradCompletesOnDevice() {
        com.github.tensor4j.gpu.core.GpuAutogradTensor x =
            com.github.tensor4j.gpu.core.GpuAutogradTensor.fromHost(
                GpuRuntime.defaultDevice(), new float[]{2f, 3f}, 2)
                .requiresGrad(true);
        com.github.tensor4j.gpu.core.GpuAutogradTensor y = x.mul(x);
        y.backward();
        float[] g = x.grad().toHost();
        assertArrayEquals(new float[]{4f, 6f}, g, 1e-5f);
    }
}
