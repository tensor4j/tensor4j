/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.gpu.core;

/**
 * Typed kernel launch handle.
 * Arguments are type-checked at launch against the compiled kernel signature.
 */
public interface GpuKernel extends AutoCloseable {

    int numArgs();

    void launch(GpuStream stream, int gridX, int gridY, int gridZ,
                int blockX, int blockY, int blockZ,
                GpuBuffer[] buffers, int[] intArgs, float[] floatArgs);

    void close();
}
