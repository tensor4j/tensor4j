/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.gpu.hip;

import java.lang.ref.Cleaner;
import com.github.tensor4j.gpu.core.GpuBuffer;
import com.github.tensor4j.gpu.core.GpuStream;

/**
 * HIP device memory with auto-cleanup.
 */
public final class HipBuffer implements GpuBuffer {

    private static final Cleaner CLEANER = Cleaner.create();

    private final long devPtr;
    private final long bytes;
    private volatile boolean closed;
    private final Cleaner.Cleanable cleanable;

    HipBuffer(long devPtr, long bytes, boolean registerCleaner) {
        this.devPtr = devPtr;
        this.bytes = bytes;
        if (registerCleaner) {
            this.cleanable = CLEANER.register(this, () -> HipDevice.nMemFree(devPtr));
        } else {
            this.cleanable = null;
        }
    }

    public long devicePointer() {
        return devPtr;
    }

    @Override
    public long bytes() { return bytes; }

    @Override
    public int numel() { return Math.toIntExact(bytes / 4L); }

    @Override
    public void copyFromHost(float[] src, long srcOffset, long dstOffset, long count) {
    }

    @Override
    public void copyFromHostAsync(float[] src, long srcOffset, long dstOffset, long count, GpuStream stream) {
    }

    @Override
    public void copyToHost(float[] dst, long srcOffset, long dstOffset, long count) {
    }

    @Override
    public void copyToHostAsync(float[] dst, long srcOffset, long dstOffset, long count, GpuStream stream) {
    }

    @Override
    public void copyFromDevice(GpuBuffer src, long srcOff, long dstOff, long bytes, GpuStream stream) {
    }

    @Override
    public void memset(byte val) {
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            if (cleanable != null) {
                cleanable.clean();
            }
        }
    }
}
