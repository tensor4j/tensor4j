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

import com.github.tensor4j.gpu.core.*;

/**
 * OpenCL device — native bridge to any OpenCL 1.2+ runtime
 * (NVIDIA, AMD, Intel, ARM). Loads {@code libtensor4j-opencl.so}.
 */
public final class OclDevice implements GpuDevice {

    private static boolean nativeLoaded;

    private final int platformIndex;
    private final int deviceIndex;
    private final String deviceName;
    private final long totalMem;
    private boolean closed;

    public OclDevice() {
        this(0, 0);
    }

    public OclDevice(int platformIndex, int deviceIndex) {
        ensureLoaded();
        this.platformIndex = platformIndex;
        this.deviceIndex = deviceIndex;
        this.deviceName = nGetName(platformIndex, deviceIndex);
        this.totalMem = nGetTotalMemory(platformIndex, deviceIndex);
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
        return nGetFreeMemory(platformIndex, deviceIndex);
    }

    @Override
    public GpuStream createStream() {
        ensureNotClosed();
        long queuePtr = nCreateCommandQueue(platformIndex, deviceIndex);
        return new OclStream(queuePtr);
    }

    @Override
    public GpuBuffer allocate(long bytes) {
        ensureNotClosed();
        long memPtr = nCreateBuffer(platformIndex, deviceIndex, bytes);
        return new OclBuffer(memPtr, bytes, true);
    }

    @Override
    public GpuBuffer allocateManaged(long bytes) {
        return allocate(bytes);
    }

    @Override
    public GpuProgram compile(String source, String kernelName) {
        ensureNotClosed();
        long programPtr = nBuildProgram(platformIndex, deviceIndex, source);
        return new OclProgram(programPtr);
    }

    @Override
    public void synchronize() {
        ensureNotClosed();
        nFinish(platformIndex, deviceIndex);
    }

    @Override
    public void close() {
        closed = true;
    }

    public int platformIndex() {
        return platformIndex;
    }

    public int deviceIndex() {
        return deviceIndex;
    }

    private void ensureNotClosed() {
        if (closed) {
            throw new IllegalStateException("device is closed");
        }
    }

    private static void ensureLoaded() {
        if (!nativeLoaded) {
            synchronized (OclDevice.class) {
                if (!nativeLoaded) {
                    try {
                        System.loadLibrary("tensor4j-opencl");
                    } catch (UnsatisfiedLinkError e) {
                        throw new UnsatisfiedLinkError(
                            "libtensor4j-opencl.so not found on java.library.path. " +
                            "Build it from native/tensor4j-opencl/.");
                    }
                    nativeLoaded = true;
                }
            }
        }
    }

    // --- native methods ---

    static native String nGetName(int platform, int device);
    static native long nGetTotalMemory(int platform, int device);
    static native long nGetFreeMemory(int platform, int device);
    static native long nCreateCommandQueue(int platform, int device);
    static native long nCreateBuffer(int platform, int device, long bytes);
    static native void nReleaseBuffer(long memPtr);
    static native long nBuildProgram(int platform, int device, String source);
    static native long nCreateKernel(long programPtr, String name);
    static native void nSetKernelArg(long kernelPtr, int index, long bufferPtr);
    static native void nSetKernelArgInt(long kernelPtr, int index, int value);
    static native void nSetKernelArgFloat(long kernelPtr, int index, float value);
    static native void nEnqueueNDRange(long queuePtr, long kernelPtr,
                                                int dims, long[] globalWork, long[] localWork);
    static native void nEnqueueWriteBuffer(long queuePtr, long memPtr,
                                                     float[] src, long srcOff, long count);
    static native void nEnqueueReadBuffer(long queuePtr, long memPtr,
                                                    float[] dst, long dstOff, long count);
    static native void nFinish(int platform, int device);
    static native void nReleaseCommandQueue(long queuePtr);
}
