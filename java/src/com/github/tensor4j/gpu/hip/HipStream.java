/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.gpu.hip;

import com.github.tensor4j.gpu.core.GpuEvent;
import com.github.tensor4j.gpu.core.GpuStream;

/**
 * HIP stream wrapper.
 */
public final class HipStream implements GpuStream {

    private final long streamPtr;
    private boolean closed;

    HipStream(long streamPtr) {
        this.streamPtr = streamPtr;
    }

    @Override
    public void synchronize() {
        HipDevice.nStreamSynchronize(streamPtr);
    }

    @Override
    public void waitEvent(GpuEvent event) {
    }

    @Override
    public GpuEvent createEvent() {
        return null;
    }

    @Override
    public void recordEvent(GpuEvent event) {
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            HipDevice.nStreamDestroy(streamPtr);
        }
    }
}
