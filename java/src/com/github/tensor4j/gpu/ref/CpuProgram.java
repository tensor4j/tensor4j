/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.gpu.ref;

import com.github.tensor4j.gpu.core.GpuKernel;
import com.github.tensor4j.gpu.core.GpuProgram;

/**
 * No-op program — CPU reference ignores compile requests
 * and returns a stub kernel.
 */
public final class CpuProgram implements GpuProgram {

    private final String kernelName;

    public CpuProgram(String kernelName) {
        this.kernelName = kernelName;
    }

    @Override
    public GpuKernel createKernel(String name) {
        return new CpuKernel(kernelName + "::" + name);
    }

    @Override
    public void close() {
    }
}
