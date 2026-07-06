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
import com.github.tensor4j.gpu.core.GpuKernel;
import com.github.tensor4j.gpu.core.GpuProgram;

/**
 * Compiled CUDA module (PTX loaded via {@code cuModuleLoadData}).
 */
public final class CudaProgram implements GpuProgram {

    private final long modulePtr;
    private final Map<String, CudaKernel> kernels;

    CudaProgram(long modulePtr) {
        this.modulePtr = modulePtr;
        this.kernels = new HashMap<>();
    }

    @Override
    public GpuKernel createKernel(String name) {
        CudaKernel k = kernels.get(name);
        if (k == null) {
            long funcPtr = CudaDevice.nModuleGetFunction(modulePtr, name);
            k = new CudaKernel(funcPtr);
            kernels.put(name, k);
        }
        return k;
    }

    @Override
    public void close() {
    }
}
