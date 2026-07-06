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

import com.github.tensor4j.gpu.core.GpuBuffer;
import com.github.tensor4j.gpu.core.GpuKernel;
import com.github.tensor4j.gpu.core.GpuStream;

/**
 * OpenCL kernel ({@code cl_kernel}) — wraps kernel handle.
 */
public final class OclKernel implements GpuKernel {

    private final long kernelPtr;

    OclKernel(long kernelPtr) {
        this.kernelPtr = kernelPtr;
    }

    @Override
    public int numArgs() {
        return 0;
    }

    @Override
    public void launch(GpuStream stream, int gridX, int gridY, int gridZ,
                       int blockX, int blockY, int blockZ,
                       GpuBuffer[] buffers, int[] intArgs, float[] floatArgs) {
        int argIndex = 0;
        for (GpuBuffer buf : buffers) {
            OclBuffer ob = (OclBuffer) buf;
            OclDevice.nSetKernelArg(kernelPtr, argIndex++, ob.devicePointer());
        }
        for (int v : intArgs) {
            OclDevice.nSetKernelArgInt(kernelPtr, argIndex++, v);
        }
        for (float v : floatArgs) {
            OclDevice.nSetKernelArgFloat(kernelPtr, argIndex++, v);
        }
        long queuePtr = ((OclStream) stream).queuePtr();
        long[] globalWork = new long[]{(long) gridX * blockX, gridY, gridZ};
        long[] localWork = new long[]{blockX, blockY, blockZ};
        OclDevice.nEnqueueNDRange(queuePtr, kernelPtr, 1, globalWork, localWork);
    }

    @Override
    public void close() {
    }
}
