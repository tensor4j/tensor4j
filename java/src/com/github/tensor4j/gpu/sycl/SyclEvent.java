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

public final class SyclEvent implements GpuEvent {

    private final long eventPtr;
    private boolean closed;

    SyclEvent(long eventPtr) {
        this.eventPtr = eventPtr;
    }

    long eventPtr() {
        return eventPtr;
    }

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
        if (!closed) {
            closed = true;
            SyclDevice.nEventDestroy(eventPtr);
        }
    }
}
