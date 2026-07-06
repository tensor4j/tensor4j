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

import java.util.HashMap;
import java.util.Map;
import com.github.tensor4j.gpu.core.*;

/**
 * CUDA device — native bridge to NVIDIA GPU via tensor4j's own JNI layer.
 * <p>
 * On first use, loads {@code libtensor4j-cuda.so} (built from
 * {@code native/tensor4j-cuda/}). If the native library is unavailable,
 * all operations throw {@link UnsatisfiedLinkError}.
 */
public final class CudaDevice implements GpuDevice {

    private static boolean nativeLoaded;

    private final int deviceOrdinal;
    private final String deviceName;
    private final int ccMajor;
    private final int ccMinor;
    private final long totalMem;
    private boolean closed;

    public CudaDevice() {
        this(0);
    }

    public CudaDevice(int ordinal) {
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
        return new CudaStream(ptr);
    }

    @Override
    public GpuBuffer allocate(long bytes) {
        ensureNotClosed();
        if (bytes <= 0) {
            throw new IllegalArgumentException("bytes must be positive: " + bytes);
        }
        long devPtr = nMemAlloc(bytes);
        return new CudaBuffer(devPtr, bytes, true);
    }

    @Override
    public GpuBuffer allocateManaged(long bytes) {
        ensureNotClosed();
        if (bytes <= 0) {
            throw new IllegalArgumentException("bytes must be positive: " + bytes);
        }
        long devPtr = nMemAllocManaged(bytes);
        return new CudaBuffer(devPtr, bytes, true);
    }

    @Override
    public GpuProgram compile(String source, String kernelName) {
        ensureNotClosed();
        long modulePtr = nCompile(source);
        return new CudaProgram(modulePtr);
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
            synchronized (CudaDevice.class) {
                if (!nativeLoaded) {
                    try {
                        System.loadLibrary("tensor4j-cuda");
                    } catch (UnsatisfiedLinkError e) {
                        throw new UnsatisfiedLinkError(
                            "libtensor4j-cuda.so not found on java.library.path. " +
                            "Build it from native/tensor4j-cuda/ or install the tensor4j-cuda package.");
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
    static native long nMemAllocManaged(long bytes);
    static native void nMemFree(long devPtr);
    static native void nMemcpyHtoD(long dstDevPtr, float[] src, long srcOff, long count);
    static native void nMemcpyDtoH(float[] dst, long dstOff, long srcDevPtr, long srcOff, long count);
    static native void nMemcpyDtoD(long dstDevPtr, long dstOff, long srcDevPtr, long srcOff, long bytes);
    static native void nMemset(long devPtr, byte val, long bytes);
    static native long nCompile(String source);
    static native long nModuleGetFunction(long modulePtr, String name);
    static native void nLaunchKernel(long funcPtr, int gridX, int gridY, int gridZ,
                                              int blockX, int blockY, int blockZ,
                                              long[] bufferPtrs, int[] intArgs, float[] floatArgs);
    static native long nStreamCreate();
    static native void nStreamSynchronize(long streamPtr);
    static native void nStreamDestroy(long streamPtr);
    static native void nDeviceSynchronize();
}
