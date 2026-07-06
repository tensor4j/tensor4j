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
 * Integration test for HIP backend native library loading.
 * Requires libtensor4j-hip.so on java.library.path.
 */
@Tag("gpu-it")
class HipDeviceIT {

    @Test
    void hipDeviceClassLoads() {
        assertDoesNotThrow(() -> Class.forName("com.github.tensor4j.gpu.hip.HipDevice"));
    }

    @Test
    void hipDeviceImplementsGpuDevice() {
        assertTrue(
            com.github.tensor4j.gpu.core.GpuDevice.class
                .isAssignableFrom(com.github.tensor4j.gpu.hip.HipDevice.class));
    }

    @Test
    void hipDeviceInstantiationSucceedsWithNativeLib() throws Exception {
        try {
            Object dev = Class.forName("com.github.tensor4j.gpu.hip.HipDevice")
                .getDeclaredConstructor()
                .newInstance();
            assertNotNull(dev);
        } catch (java.lang.reflect.InvocationTargetException e) {
            Throwable cause = e.getCause();
            if (cause instanceof UnsatisfiedLinkError) {
                throw (UnsatisfiedLinkError) cause;
            }
            throw new RuntimeException("unexpected error", cause);
        }
    }

    @Test
    void hipDeviceReportsMemory() throws Exception {
        Object dev = Class.forName("com.github.tensor4j.gpu.hip.HipDevice")
            .getDeclaredConstructor()
            .newInstance();
        String name = (String) dev.getClass().getMethod("name").invoke(dev);
        long totalMem = (long) dev.getClass().getMethod("totalMemoryBytes").invoke(dev);
        assertNotNull(name);
        assertTrue(totalMem > 0);
        dev.getClass().getMethod("close").invoke(dev);
    }
}
