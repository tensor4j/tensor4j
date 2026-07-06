/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.gpu.opencl;

import java.lang.ref.Cleaner;
import com.github.tensor4j.gpu.core.GpuBuffer;
import com.github.tensor4j.gpu.core.GpuStream;

/**
 * OpenCL device memory ({@code cl_mem}) with automatic clean-up.
 */
public final class OclBuffer implements GpuBuffer {

    private static final Cleaner CLEANER = Cleaner.create();

    private final long memPtr;
    private final long bytes;
    private volatile boolean closed;
    private final Cleaner.Cleanable cleanable;

    OclBuffer(long memPtr, long bytes, boolean registerCleaner) {
        this.memPtr = memPtr;
        this.bytes = bytes;
        if (registerCleaner) {
            this.cleanable = CLEANER.register(this, new NativeRelease(memPtr));
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
        return memPtr;
    }

    @Override
    public void copyFromHost(float[] src, long srcOffset, long dstOffset, long count) {
        checkNotClosed();
        OclDevice.nEnqueueWriteBuffer(0L, memPtr, src, srcOffset, count);
    }

    @Override
    public void copyFromHostAsync(float[] src, long srcOffset, long dstOffset, long count, GpuStream stream) {
        checkNotClosed();
        OclDevice.nEnqueueWriteBuffer(((OclStream) stream).queuePtr(), memPtr, src, srcOffset, count);
    }

    @Override
    public void copyToHost(float[] dst, long srcOffset, long dstOffset, long count) {
        checkNotClosed();
        OclDevice.nEnqueueReadBuffer(0L, memPtr, dst, dstOffset, count);
    }

    @Override
    public void copyToHostAsync(float[] dst, long srcOffset, long dstOffset, long count, GpuStream stream) {
        checkNotClosed();
        OclDevice.nEnqueueReadBuffer(((OclStream) stream).queuePtr(), memPtr, dst, dstOffset, count);
    }

    @Override
    public void copyFromDevice(GpuBuffer src, long srcOff, long dstOff, long bytes, GpuStream stream) {
        checkNotClosed();
    }

    @Override
    public void memset(byte val) {
        checkNotClosed();
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

    private static class NativeRelease implements Runnable {
        private final long ptr;
        NativeRelease(long ptr) { this.ptr = ptr; }
        @Override
        public void run() {
            try {
                OclDevice.nReleaseBuffer(ptr);
            } catch (Exception ignored) {
            }
        }
    }
}
