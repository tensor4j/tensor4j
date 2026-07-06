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

import com.github.tensor4j.gpu.core.GpuBuffer;
import com.github.tensor4j.gpu.core.GpuKernel;
import com.github.tensor4j.gpu.core.GpuStream;

public final class SyclKernel implements GpuKernel {

    private final long kernelPtr;
    private final int numArgs;
    private boolean closed;

    SyclKernel(long kernelPtr, int numArgs) {
        this.kernelPtr = kernelPtr;
        this.numArgs = numArgs;
    }

    @Override
    public int numArgs() {
        return numArgs;
    }

    @Override
    public void launch(GpuStream stream, int gridX, int gridY, int gridZ,
                       int blockX, int blockY, int blockZ,
                       GpuBuffer[] buffers, int[] intArgs, float[] floatArgs) {
        long[] bufPtrs = new long[buffers.length];
        for (int i = 0; i < buffers.length; i++) {
            bufPtrs[i] = ((SyclBuffer) buffers[i]).devicePointer();
        }
        long queuePtr = ((SyclStream) stream).queuePtr();
        SyclDevice.nLaunchKernel(kernelPtr, queuePtr,
                                 gridX, gridY, gridZ,
                                 blockX, blockY, blockZ,
                                 bufPtrs, intArgs, floatArgs);
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            SyclDevice.nReleaseKernel(kernelPtr);
        }
    }
}
