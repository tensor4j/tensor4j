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

import com.github.tensor4j.gpu.core.GpuBuffer;
import com.github.tensor4j.gpu.core.GpuKernel;
import com.github.tensor4j.gpu.core.GpuStream;

/**
 * CPU reference kernel — launchers are no-ops (CPU device
 * executes synchronously via direct buffer manipulation).
 */
public final class CpuKernel implements GpuKernel {

    private final String name;

    public CpuKernel(String name) {
        this.name = name;
    }

    @Override
    public int numArgs() {
        return 0;
    }

    @Override
    public void launch(GpuStream stream, int gridX, int gridY, int gridZ,
                       int blockX, int blockY, int blockZ,
                       GpuBuffer[] buffers, int[] intArgs, float[] floatArgs) {
    }

    @Override
    public void close() {
    }

    public String name() {
        return name;
    }
}
