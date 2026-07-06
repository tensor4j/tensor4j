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

import java.util.HashMap;
import java.util.Map;
import com.github.tensor4j.gpu.core.GpuKernel;
import com.github.tensor4j.gpu.core.GpuProgram;

/**
 * Compiled OpenCL program ({@code cl_program}).
 */
public final class OclProgram implements GpuProgram {

    private final long programPtr;
    private final Map<String, OclKernel> kernels;

    OclProgram(long programPtr) {
        this.programPtr = programPtr;
        this.kernels = new HashMap<>();
    }

    @Override
    public GpuKernel createKernel(String name) {
        OclKernel k = kernels.get(name);
        if (k == null) {
            long kernelPtr = OclDevice.nCreateKernel(programPtr, name);
            k = new OclKernel(kernelPtr);
            kernels.put(name, k);
        }
        return k;
    }

    @Override
    public void close() {
    }
}
