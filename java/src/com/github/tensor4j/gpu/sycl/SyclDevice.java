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

import com.github.tensor4j.gpu.core.*;

/**
 * SYCL device — native bridge to any SYCL-capable GPU via oneAPI DPC++,
 * AdaptiveCpp (hipSYCL), or ComputeCpp.
 * <p>
 * On first use, loads {@code libtensor4j-sycl.so} (built from
 * {@code native/tensor4j-sycl/}). If the native library is unavailable,
 * all operations throw {@link UnsatisfiedLinkError}.
 */
public final class SyclDevice implements GpuDevice {

    private static boolean nativeLoaded;

    private final int deviceOrdinal;
    private final String deviceName;
    private final int ccMajor;
    private final int ccMinor;
    private final long totalMem;
    private boolean closed;

    public SyclDevice() {
        this(0);
    }

    public SyclDevice(int ordinal) {
        ensureLoaded();
        this.deviceOrdinal = ordinal;
        this.deviceName = nGetName(ordinal);
        this.ccMajor = nGetComputeCapabilityMajor(ordinal);
        this.ccMinor = nGetComputeCapabilityMinor(ordinal);
        this.totalMem = nGetTotalMemory(ordinal);
    }

    @Override
    public String name() {
        return deviceName;
    }

    @Override
    public int computeCapabilityMajor() {
        return ccMajor;
    }

    @Override
    public int computeCapabilityMinor() {
        return ccMinor;
    }

    @Override
    public long totalMemoryBytes() {
        return totalMem;
    }

    @Override
    public long freeMemoryBytes() {
        ensureNotClosed();
        return nGetFreeMemory(deviceOrdinal);
    }

    @Override
    public GpuStream createStream() {
        ensureNotClosed();
        long ptr = nStreamCreate();
        return new SyclStream(ptr);
    }

    @Override
    public GpuBuffer allocate(long bytes) {
        ensureNotClosed();
        if (bytes <= 0) {
            throw new IllegalArgumentException("bytes must be positive: " + bytes);
        }
        long devPtr = nMemAlloc(bytes);
        return new SyclBuffer(devPtr, bytes, true);
    }

    @Override
    public GpuBuffer allocateManaged(long bytes) {
        ensureNotClosed();
        if (bytes <= 0) {
            throw new IllegalArgumentException("bytes must be positive: " + bytes);
        }
        long devPtr = nMemAllocShared(bytes);
        return new SyclBuffer(devPtr, bytes, true);
    }

    @Override
    public GpuProgram compile(String source, String kernelName) {
        ensureNotClosed();
        long programPtr = nCompile(source);
        return new SyclProgram(programPtr);
    }

    @Override
    public void synchronize() {
        ensureNotClosed();
        nDeviceSynchronize();
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
        }
    }

    public int ordinal() {
        return deviceOrdinal;
    }

    private void ensureNotClosed() {
        if (closed) {
            throw new IllegalStateException("device is closed");
        }
    }

    private static void ensureLoaded() {
        if (!nativeLoaded) {
            synchronized (SyclDevice.class) {
                if (!nativeLoaded) {
                    try {
                        System.loadLibrary("tensor4j-sycl");
                    } catch (UnsatisfiedLinkError e) {
                        throw new UnsatisfiedLinkError(
                            "libtensor4j-sycl.so not found on java.library.path. " +
                            "Build it from native/tensor4j-sycl/ or install the tensor4j-sycl package.");
                    }
                    nativeLoaded = true;
                }
            }
        }
    }

    // --- native methods ---

    static native String nGetName(int ordinal);
    static native int nGetComputeCapabilityMajor(int ordinal);
    static native int nGetComputeCapabilityMinor(int ordinal);
    static native long nGetTotalMemory(int ordinal);
    static native long nGetFreeMemory(int ordinal);
    static native long nMemAlloc(long bytes);
    static native long nMemAllocShared(long bytes);
    static native void nMemFree(long devPtr);
    static native void nMemcpyHtoD(long dstDevPtr, float[] src, long srcOff, long dstOff, long count);
    static native void nMemcpyHtoDAsync(long dstDevPtr, float[] src, long srcOff, long dstOff, long count, long queuePtr);
    static native void nMemcpyDtoH(float[] dst, long dstOff, long srcDevPtr, long srcOff, long count);
    static native void nMemcpyDtoHAsync(float[] dst, long dstOff, long srcDevPtr, long srcOff, long count, long queuePtr);
    static native void nMemcpyDtoD(long dstDevPtr, long dstOff, long srcDevPtr, long srcOff, long bytes, long queuePtr);
    static native void nMemset(long devPtr, byte val, long bytes);
    static native long nCompile(String source);
    static native long nCreateKernel(long programPtr, String name);
    static native void nLaunchKernel(long kernelPtr, long queuePtr,
                                     int gridX, int gridY, int gridZ,
                                     int blockX, int blockY, int blockZ,
                                     long[] bufferPtrs, int[] intArgs, float[] floatArgs);
    static native long nStreamCreate();
    static native void nStreamSynchronize(long queuePtr);
    static native void nStreamDestroy(long queuePtr);
    static native void nDeviceSynchronize();
    static native long nEventCreate();
    static native void nEventDestroy(long eventPtr);
    static native void nReleaseKernel(long kernelPtr);
    static native void nReleaseProgram(long programPtr);
}
