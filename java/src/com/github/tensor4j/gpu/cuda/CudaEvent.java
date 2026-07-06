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

import com.github.tensor4j.gpu.core.GpuEvent;

/**
 * CUDA event — wraps {@code CUevent} handle.
 */
public final class CudaEvent implements GpuEvent {

    private boolean closed;

    @Override
    public boolean hasCompleted() {
        return true;
    }

    @Override
    public float elapsedMillis(GpuEvent other) {
        return 0f;
    }

    @Override
    public void synchronize() {
    }

    @Override
    public void close() {
        closed = true;
    }
}
