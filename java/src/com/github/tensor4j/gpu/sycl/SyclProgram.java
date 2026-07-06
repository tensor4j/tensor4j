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

import java.util.HashMap;
import java.util.Map;
import com.github.tensor4j.gpu.core.GpuKernel;
import com.github.tensor4j.gpu.core.GpuProgram;

public final class SyclProgram implements GpuProgram {

    private final long programPtr;
    private final Map<String, SyclKernel> kernels = new HashMap<>();
    private boolean closed;

    SyclProgram(long programPtr) {
        this.programPtr = programPtr;
    }

    @Override
    public GpuKernel createKernel(String name) {
        if (closed) {
            throw new IllegalStateException("program is closed");
        }
        return kernels.computeIfAbsent(name, k -> {
            long kernelPtr = SyclDevice.nCreateKernel(programPtr, name);
            return new SyclKernel(kernelPtr, 0);
        });
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            for (SyclKernel k : kernels.values()) {
                k.close();
            }
            kernels.clear();
            SyclDevice.nReleaseProgram(programPtr);
        }
    }
}
