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

/**
 * Integration test for CUDA backend native library loading.
 * Requires libtensor4j-cuda.so on java.library.path.
 */
@Tag("gpu-it")
class CudaDeviceIT {

    @Test
    void cudaDeviceClassLoads() {
        assertDoesNotThrow(() -> Class.forName("com.github.tensor4j.gpu.cuda.CudaDevice"));
    }

    @Test
    void cudaDeviceImplementsGpuDevice() {
        assertTrue(
            com.github.tensor4j.gpu.core.GpuDevice.class
                .isAssignableFrom(com.github.tensor4j.gpu.cuda.CudaDevice.class));
    }

    @Test
    void cudaDeviceInstantiationSucceedsWithNativeLib() throws Exception {
        // When libtensor4j-cuda.so is on java.library.path and CUDA is available,
        // this should succeed. Otherwise it throws UnsatisfiedLinkError.
        try {
            Object dev = Class.forName("com.github.tensor4j.gpu.cuda.CudaDevice")
                .getDeclaredConstructor()
                .newInstance();
            assertNotNull(dev);
            String name = (String) dev.getClass().getMethod("name").invoke(dev);
            assertNotNull(name);
            assertFalse(name.isEmpty());
        } catch (java.lang.reflect.InvocationTargetException e) {
            Throwable cause = e.getCause();
            if (cause instanceof UnsatisfiedLinkError) {
                throw (UnsatisfiedLinkError) cause;
            }
            throw new RuntimeException("unexpected error", cause);
        }
    }

    @Test
    void cudaDeviceReportsMetadata() throws Exception {
        Object dev = Class.forName("com.github.tensor4j.gpu.cuda.CudaDevice")
            .getDeclaredConstructor()
            .newInstance();
        String name = (String) dev.getClass().getMethod("name").invoke(dev);
        int ccMajor = (int) dev.getClass().getMethod("computeCapabilityMajor").invoke(dev);
        long totalMem = (long) dev.getClass().getMethod("totalMemoryBytes").invoke(dev);
        assertNotNull(name);
        assertTrue(ccMajor >= 2, "CUDA compute capability should be >= 2.0");
        assertTrue(totalMem > 0, "CUDA device should have > 0 total memory");
        dev.getClass().getMethod("close").invoke(dev);
    }
}
