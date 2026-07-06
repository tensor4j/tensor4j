/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.gpu.sycl;

import java.lang.ref.Cleaner;
import com.github.tensor4j.gpu.core.GpuBuffer;
import com.github.tensor4j.gpu.core.GpuStream;

public final class SyclBuffer implements GpuBuffer {

    private static final Cleaner CLEANER = Cleaner.create();

    private final long devPtr;
    private final long byteSize;
    private final Cleaner.Cleanable cleanable;
    private boolean closed;

    SyclBuffer(long devPtr, long byteSize, boolean registerCleaner) {
        this.devPtr = devPtr;
        this.byteSize = byteSize;
        this.cleanable = registerCleaner
            ? CLEANER.register(this, new NativeFree(devPtr))
            : null;
    }

    long devicePointer() {
        return devPtr;
    }

    @Override
    public long bytes() {
        return byteSize;
    }

    @Override
    public int numel() {
        return (int) (byteSize / 4L);
    }

    @Override
    public void copyFromHost(float[] src, long srcOffset, long dstOffset, long count) {
        if (closed) throw new IllegalStateException("buffer is closed");
        SyclDevice.nMemcpyHtoD(devPtr, src, srcOffset, dstOffset, count);
    }

    @Override
    public void copyFromHostAsync(float[] src, long srcOffset, long dstOffset, long count, GpuStream stream) {
        if (closed) throw new IllegalStateException("buffer is closed");
        long queuePtr = ((SyclStream) stream).queuePtr();
        SyclDevice.nMemcpyHtoDAsync(devPtr, src, srcOffset, dstOffset, count, queuePtr);
    }

    @Override
    public void copyToHost(float[] dst, long srcOffset, long dstOffset, long count) {
        if (closed) throw new IllegalStateException("buffer is closed");
        SyclDevice.nMemcpyDtoH(dst, srcOffset, devPtr, dstOffset, count);
    }

    @Override
    public void copyToHostAsync(float[] dst, long srcOffset, long dstOffset, long count, GpuStream stream) {
        if (closed) throw new IllegalStateException("buffer is closed");
        long queuePtr = ((SyclStream) stream).queuePtr();
        SyclDevice.nMemcpyDtoHAsync(dst, srcOffset, devPtr, dstOffset, count, queuePtr);
    }

    @Override
    public void copyFromDevice(GpuBuffer src, long srcOff, long dstOff, long bytes, GpuStream stream) {
        if (closed) throw new IllegalStateException("buffer is closed");
        long srcPtr = ((SyclBuffer) src).devicePointer();
        long queuePtr = ((SyclStream) stream).queuePtr();
        SyclDevice.nMemcpyDtoD(devPtr, dstOff, srcPtr, srcOff, bytes, queuePtr);
    }

    @Override
    public void memset(byte val) {
        if (closed) throw new IllegalStateException("buffer is closed");
        SyclDevice.nMemset(devPtr, val, byteSize);
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

    private static class NativeFree implements Runnable {
        private final long devPtr;
        NativeFree(long devPtr) {
            this.devPtr = devPtr;
        }
        @Override
        public void run() {
            if (devPtr != 0) {
                SyclDevice.nMemFree(devPtr);
            }
        }
    }
}
