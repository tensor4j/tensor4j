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
 * Asynchronous execution stream for GPU commands.
 * Commands enqueued on the same stream execute in order;
 * commands on different streams may overlap.
 */
public interface GpuStream extends AutoCloseable {

    void synchronize();

    void waitEvent(GpuEvent event);

    GpuEvent createEvent();

    void recordEvent(GpuEvent event);

    void close();
}
