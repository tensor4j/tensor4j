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

import com.github.tensor4j.gpu.core.GpuEvent;
import com.github.tensor4j.gpu.core.GpuStream;

public final class SyclStream implements GpuStream {

    private final long queuePtr;
    private boolean closed;

    SyclStream(long queuePtr) {
        this.queuePtr = queuePtr;
    }

    long queuePtr() {
        return queuePtr;
    }

    @Override
    public void synchronize() {
        SyclDevice.nStreamSynchronize(queuePtr);
    }

    @Override
    public void waitEvent(GpuEvent event) {
    }

    @Override
    public GpuEvent createEvent() {
        long ptr = SyclDevice.nEventCreate();
        return new SyclEvent(ptr);
    }

    @Override
    public void recordEvent(GpuEvent event) {
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            SyclDevice.nStreamDestroy(queuePtr);
        }
    }
}
