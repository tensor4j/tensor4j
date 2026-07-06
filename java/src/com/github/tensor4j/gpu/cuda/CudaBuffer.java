/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.gpu.cuda;

import java.lang.ref.Cleaner;
import com.github.tensor4j.gpu.core.GpuBuffer;
import com.github.tensor4j.gpu.core.GpuStream;

/**
 * CUDA device memory with automatic clean-up via {@link Cleaner}.
 */
public final class CudaBuffer implements GpuBuffer {

    private static final Cleaner CLEANER = Cleaner.create();

    private final long devPtr;
    private final long bytes;
    private volatile boolean closed;

    private final Cleaner.Cleanable cleanable;

    CudaBuffer(long devPtr, long bytes, boolean registerCleaner) {
        this.devPtr = devPtr;
        this.bytes = bytes;
        if (registerCleaner) {
            this.cleanable = CLEANER.register(this, new NativeFree(devPtr));
        } else {
            this.cleanable = null;
        }
    }

    @Override
    public long bytes() {
        return bytes;
    }

    @Override
    public int numel() {
        return Math.toIntExact(bytes / 4L);
    }

    public long devicePointer() {
        return devPtr;
    }

    @Override
    public void copyFromHost(float[] src, long srcOffset, long dstOffset, long count) {
        checkNotClosed();
        CudaDevice.nMemcpyHtoD(devPtr + dstOffset * 4L, src, srcOffset, count);
    }

    @Override
    public void copyFromHostAsync(float[] src, long srcOffset, long dstOffset, long count, GpuStream stream) {
        checkNotClosed();
        copyFromHost(src, srcOffset, dstOffset, count);
    }

    @Override
    public void copyToHost(float[] dst, long srcOffset, long dstOffset, long count) {
        checkNotClosed();
        CudaDevice.nMemcpyDtoH(dst, dstOffset, devPtr, srcOffset, count);
    }

    @Override
    public void copyToHostAsync(float[] dst, long srcOffset, long dstOffset, long count, GpuStream stream) {
        checkNotClosed();
        copyToHost(dst, srcOffset, dstOffset, count);
    }

    @Override
    public void copyFromDevice(GpuBuffer src, long srcOff, long dstOff, long bytes, GpuStream stream) {
        checkNotClosed();
        CudaBuffer other = (CudaBuffer) src;
        CudaDevice.nMemcpyDtoD(devPtr, dstOff, other.devPtr, srcOff, bytes);
    }

    @Override
    public void memset(byte val) {
        checkNotClosed();
        CudaDevice.nMemset(devPtr, val, bytes);
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

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("buffer is closed");
        }
    }

    private static class NativeFree implements Runnable {
        private final long ptr;
        NativeFree(long ptr) { this.ptr = ptr; }
        @Override
        public void run() {
            try {
                CudaDevice.nMemFree(ptr);
            } catch (Exception ignored) {
            }
        }
    }
}
