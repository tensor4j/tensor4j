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

import com.github.tensor4j.gpu.core.GpuEvent;
import com.github.tensor4j.gpu.core.GpuStream;

/**
 * OpenCL command queue ({@code cl_command_queue}) wrapper.
 */
public final class OclStream implements GpuStream {

    private final long queuePtr;
    private boolean closed;

    OclStream(long queuePtr) {
        this.queuePtr = queuePtr;
    }

    public long queuePtr() {
        return queuePtr;
    }

    @Override
    public void synchronize() {
    }

    @Override
    public void waitEvent(GpuEvent event) {
    }

    @Override
    public GpuEvent createEvent() {
        return new OclEvent();
    }

    @Override
    public void recordEvent(GpuEvent event) {
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            OclDevice.nReleaseCommandQueue(queuePtr);
        }
    }
}
