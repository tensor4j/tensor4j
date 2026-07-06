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
import com.github.tensor4j.gpu.ref.CpuDevice;

/**
 * SYCL backend structural tests.
 * <p>
 * Verifies that the SYCL Java classes compile correctly and that
 * GpuRuntime falls back to CPU when the native SYCL library is absent.
 * Full SYCL integration requires libtensor4j-sycl.so on java.library.path.
 */
class SyclDeviceTest {

    @Test
    void runtimeFallsBackToCpuWhenSyclUnavailable() {
        // On this CI environment without native SYCL, GpuRuntime should
        // detect SyclDevice's UnsatisfiedLinkError and fall back to CPU.
        assertInstanceOf(CpuDevice.class, com.github.tensor4j.gpu.core.GpuRuntime.defaultDevice());
    }

    @Test
    void syclDeviceClassLoads() {
        assertDoesNotThrow(() -> Class.forName("com.github.tensor4j.gpu.sycl.SyclDevice"));
    }

    @Test
    void syclBufferClassLoads() {
        assertDoesNotThrow(() -> Class.forName("com.github.tensor4j.gpu.sycl.SyclBuffer"));
    }

    @Test
    void syclProgramClassLoads() {
        assertDoesNotThrow(() -> Class.forName("com.github.tensor4j.gpu.sycl.SyclProgram"));
    }

    @Test
    void syclKernelClassLoads() {
        assertDoesNotThrow(() -> Class.forName("com.github.tensor4j.gpu.sycl.SyclKernel"));
    }

    @Test
    void syclStreamClassLoads() {
        assertDoesNotThrow(() -> Class.forName("com.github.tensor4j.gpu.sycl.SyclStream"));
    }

    @Test
    void syclEventClassLoads() {
        assertDoesNotThrow(() -> Class.forName("com.github.tensor4j.gpu.sycl.SyclEvent"));
    }

    @Test
    void syclDeviceInstantiationFailsWithoutNativeLib() {
        assertThrows(java.lang.reflect.InvocationTargetException.class, () ->
            Class.forName("com.github.tensor4j.gpu.sycl.SyclDevice")
                .getDeclaredConstructor()
                .newInstance());
    }

    @Test
    void syclDeviceImplementsGpuDevice() {
        assertTrue(
            com.github.tensor4j.gpu.core.GpuDevice.class
                .isAssignableFrom(com.github.tensor4j.gpu.sycl.SyclDevice.class));
    }

    @Test
    void syclBufferImplementsGpuBuffer() {
        assertTrue(
            com.github.tensor4j.gpu.core.GpuBuffer.class
                .isAssignableFrom(com.github.tensor4j.gpu.sycl.SyclBuffer.class));
    }

    @Test
    void syclStreamImplementsGpuStream() {
        assertTrue(
            com.github.tensor4j.gpu.core.GpuStream.class
                .isAssignableFrom(com.github.tensor4j.gpu.sycl.SyclStream.class));
    }

    @Test
    void syclProgramImplementsGpuProgram() {
        assertTrue(
            com.github.tensor4j.gpu.core.GpuProgram.class
                .isAssignableFrom(com.github.tensor4j.gpu.sycl.SyclProgram.class));
    }

    @Test
    void syclKernelImplementsGpuKernel() {
        assertTrue(
            com.github.tensor4j.gpu.core.GpuKernel.class
                .isAssignableFrom(com.github.tensor4j.gpu.sycl.SyclKernel.class));
    }

    @Test
    void syclEventImplementsGpuEvent() {
        assertTrue(
            com.github.tensor4j.gpu.core.GpuEvent.class
                .isAssignableFrom(com.github.tensor4j.gpu.sycl.SyclEvent.class));
    }
}
