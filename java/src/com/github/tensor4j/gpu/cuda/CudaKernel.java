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

import com.github.tensor4j.gpu.core.GpuBuffer;
import com.github.tensor4j.gpu.core.GpuKernel;
import com.github.tensor4j.gpu.core.GpuStream;

/**
 * CUDA kernel — wraps {@code CUfunction} handle from {@code cuModuleGetFunction}.
 */
public final class CudaKernel implements GpuKernel {

    private final long funcPtr;

    CudaKernel(long funcPtr) {
        this.funcPtr = funcPtr;
    }

    @Override
    public int numArgs() {
        return 0;
    }

    @Override
    public void launch(GpuStream stream, int gridX, int gridY, int gridZ,
                       int blockX, int blockY, int blockZ,
                       GpuBuffer[] buffers, int[] intArgs, float[] floatArgs) {
        long[] ptrs = new long[buffers.length];
        for (int i = 0; i < buffers.length; i++) {
            ptrs[i] = ((CudaBuffer) buffers[i]).devicePointer();
        }
        CudaDevice.nLaunchKernel(funcPtr, gridX, gridY, gridZ, blockX, blockY, blockZ,
                                 ptrs, intArgs, floatArgs);
    }

    @Override
    public void close() {
    }
}
