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

import com.github.tensor4j.gpu.core.*;

/**
 * AMD ROCm/HIP device — native bridge via {@code libtensor4j-hip.so}.
 */
public final class HipDevice implements GpuDevice {

    private static boolean nativeLoaded;

    private final int deviceOrdinal;
    private final String deviceName;
    private final long totalMem;
    private boolean closed;

    public HipDevice() {
        this(0);
    }

    public HipDevice(int ordinal) {
        ensureLoaded();
        this.deviceOrdinal = ordinal;
        this.deviceName = nGetName(ordinal);
        this.totalMem = nGetTotalMemory(ordinal);
    }

    @Override
    public String name() {
        return deviceName;
    }

    @Override
    public int computeCapabilityMajor() {
        return 0;
    }

    @Override
    public int computeCapabilityMinor() {
        return 0;
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
        return new HipStream(ptr);
    }

    @Override
    public GpuBuffer allocate(long bytes) {
        ensureNotClosed();
        long devPtr = nMemAlloc(bytes);
        return new HipBuffer(devPtr, bytes, true);
    }

    @Override
    public GpuBuffer allocateManaged(long bytes) {
        ensureNotClosed();
        long devPtr = nMemAllocManaged(bytes);
        return new HipBuffer(devPtr, bytes, true);
    }

    @Override
    public GpuProgram compile(String source, String kernelName) {
        ensureNotClosed();
        long modulePtr = nCompile(source);
        return new HipProgram(modulePtr);
    }

    @Override
    public void synchronize() {
        ensureNotClosed();
        nDeviceSynchronize();
    }

    @Override
    public void close() {
        closed = true;
    }

    private void ensureNotClosed() {
        if (closed) {
            throw new IllegalStateException("device is closed");
        }
    }

    private static void ensureLoaded() {
        if (!nativeLoaded) {
            synchronized (HipDevice.class) {
                if (!nativeLoaded) {
                    try {
                        System.loadLibrary("tensor4j-hip");
                    } catch (UnsatisfiedLinkError e) {
                        throw new UnsatisfiedLinkError(
                            "libtensor4j-hip.so not found. Build from native/tensor4j-hip/.");
                    }
                    nativeLoaded = true;
                }
            }
        }
    }

    // native methods
    static native String nGetName(int ordinal);
    static native long nGetTotalMemory(int ordinal);
    static native long nGetFreeMemory(int ordinal);
    static native long nMemAlloc(long bytes);
    static native long nMemAllocManaged(long bytes);
    static native void nMemFree(long devPtr);
    static native long nCompile(String source);
    static native long nModuleGetFunction(long modulePtr, String name);
    static native void nLaunchKernel(long funcPtr, int gx, int gy, int gz,
                                              int bx, int by, int bz,
                                              long[] bufPtrs, int[] ints, float[] floats);
    static native long nStreamCreate();
    static native void nStreamSynchronize(long streamPtr);
    static native void nStreamDestroy(long streamPtr);
    static native void nDeviceSynchronize();
}
