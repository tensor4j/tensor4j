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
 * GPU core interface tests using the CPU reference backend.
 * These tests run without GPU hardware (CI-safe).
 */
class GpuDeviceTest {

    @Test
    void cpuDeviceReportsMetadata() {
        try (CpuDevice dev = new CpuDevice()) {
            assertNotNull(dev.name());
            assertEquals(0, dev.computeCapabilityMajor());
            assertTrue(dev.totalMemoryBytes() > 0);
        }
    }

    @Test
    void allocateAndValidateBuffer() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBuffer buf = dev.allocate(1024L);
            assertEquals(1024L, buf.bytes());
            assertEquals(256, buf.numel());
            buf.close();
        }
    }

    @Test
    void allocateManagedIsSameAsAllocateOnCpu() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBuffer regular = dev.allocate(256L);
            GpuBuffer managed = dev.allocateManaged(256L);
            assertEquals(regular.bytes(), managed.bytes());
            regular.close();
            managed.close();
        }
    }

    @Test
    void bufferCopyFromHostAndToHost() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBuffer buf = dev.allocate(16 * 4L);
            float[] src = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f};
            buf.copyFromHost(src, 0, 0, 16);

            float[] dst = new float[16];
            buf.copyToHost(dst, 0, 0, 16);
            assertArrayEquals(src, dst, 1e-6f);
        }
    }

    @Test
    void bufferCopyWithOffset() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBuffer buf = dev.allocate(32 * 4L);
            float[] src = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
            buf.copyFromHost(src, 2, 10, src.length - 2);

            float[] dst = new float[12];
            buf.copyToHost(dst, 10, 0, dst.length);
            assertArrayEquals(new float[]{3f, 4f, 5f, 6f, 7f, 8f, 0f, 0f, 0f, 0f, 0f, 0f}, dst, 1e-6f);
        }
    }

    @Test
    void bufferMemset() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBuffer buf = dev.allocate(8 * 4L);
            float[] src = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
            buf.copyFromHost(src, 0, 0, 8);
            buf.memset((byte) 0);

            float[] dst = new float[8];
            buf.copyToHost(dst, 0, 0, 8);
            for (float v : dst) {
                assertEquals(0f, v, 1e-6f);
            }
        }
    }

    @Test
    void bufferCloseIsIdempotent() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuBuffer buf = dev.allocate(64L);
            buf.close();
            buf.close();
            assertThrows(IllegalStateException.class, () -> buf.copyToHost(new float[1], 0, 0, 1));
        }
    }

    @Test
    void streamCreateAndUse() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuStream stream = dev.createStream();
            assertNotNull(stream);
            stream.synchronize();
            stream.close();
        }
    }

    @Test
    void streamEventLifecycle() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuStream stream = dev.createStream();
            GpuEvent event = stream.createEvent();
            assertTrue(event.hasCompleted());
            stream.recordEvent(event);
            stream.waitEvent(event);
            stream.close();
            event.close();
        }
    }

    @Test
    void compileReturnsProgram() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuProgram prog = dev.compile("__kernel void test() {}", "test");
            assertNotNull(prog);
            GpuKernel kernel = prog.createKernel("test");
            assertNotNull(kernel);
            prog.close();
        }
    }

    @Test
    void deviceSynchronize() {
        try (CpuDevice dev = new CpuDevice()) {
            dev.synchronize();
        }
    }

    @Test
    void deviceCloseIsIdempotent() {
        CpuDevice dev = new CpuDevice();
        dev.close();
        dev.close();
    }

    @Test
    void gpuTensorFromHostAndToHost() {
        try (CpuDevice dev = new CpuDevice()) {
            float[] data = {1f, 2f, 3f, 4f, 5f, 6f};
            GpuTensor t = GpuTensor.fromHost(dev, data, 2, 3);
            assertArrayEquals(new int[]{2, 3}, t.shape());
            assertEquals(6, t.numel());
            assertArrayEquals(data, t.toHost(), 1e-6f);
        }
    }

    @Test
    void gpuTensorBroadcastShape() {
        try (CpuDevice dev = new CpuDevice()) {
            float[] data = {1f, 2f, 3f, 4f};
            GpuTensor t = GpuTensor.fromHost(dev, data, 1, 4);
            assertArrayEquals(new int[]{1, 4}, t.shape());
            assertArrayEquals(new int[]{4, 1}, t.strides());
        }
    }

    @Test
    void reshapePreservesData() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensor t = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3);
            GpuTensor r = t.reshape(3, 2);
            assertArrayEquals(new int[]{3, 2}, r.shape());
            assertArrayEquals(t.toHost(), r.toHost(), 1e-6f);
        }
    }

    @Test
    void reshapeRejectsMismatchedElements() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensor t = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 3);
            assertThrows(IllegalArgumentException.class, () -> t.reshape(2, 2));
        }
    }

    @Test
    void squeezeRemovesSingletons() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensor t = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 1, 3, 1);
            GpuTensor s = t.squeeze();
            assertArrayEquals(new int[]{3}, s.shape());
            assertArrayEquals(new float[]{1f, 2f, 3f}, s.toHost(), 1e-6f);
        }
    }

    @Test
    void squeezeNoopWhenNoSingletons() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensor t = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 3);
            GpuTensor s = t.squeeze();
            assertArrayEquals(new int[]{3}, s.shape());
        }
    }

    @Test
    void unsqueezeInsertsDim() {
        try (CpuDevice dev = new CpuDevice()) {
            GpuTensor t = GpuTensor.fromHost(dev, new float[]{1f, 2f, 3f}, 3);
            GpuTensor u = t.unsqueeze(0);
            assertArrayEquals(new int[]{1, 3}, u.shape());
            u = t.unsqueeze(1);
            assertArrayEquals(new int[]{3, 1}, u.shape());
        }
    }
}
