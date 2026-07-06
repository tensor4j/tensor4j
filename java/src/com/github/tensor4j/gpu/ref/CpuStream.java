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

import com.github.tensor4j.gpu.core.GpuEvent;
import com.github.tensor4j.gpu.core.GpuStream;

/**
 * No-op stream — CPU reference executes synchronously.
 */
public final class CpuStream implements GpuStream {

    private boolean closed;

    @Override
    public void synchronize() {
    }

    @Override
    public void waitEvent(GpuEvent event) {
    }

    @Override
    public GpuEvent createEvent() {
        return new CpuEvent();
    }

    @Override
    public void recordEvent(GpuEvent event) {
    }

    @Override
    public void close() {
        closed = true;
    }
}
