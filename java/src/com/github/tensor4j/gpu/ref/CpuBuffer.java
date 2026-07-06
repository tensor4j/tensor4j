/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.gpu.ref;

import java.util.Arrays;
import com.github.tensor4j.gpu.core.GpuBuffer;
import com.github.tensor4j.gpu.core.GpuStream;

/**
 * Heap-backed GPU buffer for the CPU reference device.
 * All operations are synchronous (streams are ignored).
 */
public final class CpuBuffer implements GpuBuffer {

    private float[] data;
    private boolean closed;

    public CpuBuffer(long bytes) {
        int numel = Math.toIntExact(bytes / 4L);
        if (numel * 4L != bytes) {
            throw new IllegalArgumentException("bytes must be multiple of 4: " + bytes);
        }
        this.data = new float[numel];
    }

    @Override
    public long bytes() {
        return (long) data.length * 4L;
    }

    @Override
    public int numel() {
        return data.length;
    }

    @Override
    public void copyFromHost(float[] src, long srcOffset, long dstOffset, long count) {
        checkNotClosed();
        System.arraycopy(src, (int) srcOffset, data, (int) dstOffset, (int) count);
    }

    @Override
    public void copyFromHostAsync(float[] src, long srcOffset, long dstOffset, long count, GpuStream stream) {
        copyFromHost(src, srcOffset, dstOffset, count);
    }

    @Override
    public void copyToHost(float[] dst, long srcOffset, long dstOffset, long count) {
        checkNotClosed();
        System.arraycopy(data, (int) srcOffset, dst, (int) dstOffset, (int) count);
    }

    @Override
    public void copyToHostAsync(float[] dst, long srcOffset, long dstOffset, long count, GpuStream stream) {
        copyToHost(dst, srcOffset, dstOffset, count);
    }

    @Override
    public void copyFromDevice(GpuBuffer src, long srcOff, long dstOff, long bytes, GpuStream stream) {
        checkNotClosed();
        CpuBuffer other = (CpuBuffer) src;
        System.arraycopy(other.data, (int) (srcOff / 4L), data, (int) (dstOff / 4L), (int) (bytes / 4L));
    }

    @Override
    public void memset(byte val) {
        checkNotClosed();
        int iv = val & 0xFF;
        int pattern = iv | (iv << 8) | (iv << 16) | (iv << 24);
        Arrays.fill(data, Float.intBitsToFloat(pattern));
    }

    public float[] unsafeData() {
        return data;
    }

    @Override
    public void close() {
        closed = true;
        data = null;
    }

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("buffer is closed");
        }
    }
}
