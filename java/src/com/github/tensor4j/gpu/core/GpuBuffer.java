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
 * Device memory allocation: typed float32 operations with optional auto-free.
 */
public interface GpuBuffer extends AutoCloseable {

    long bytes();

    int numel();

    void copyFromHost(float[] src, long srcOffset, long dstOffset, long count);

    void copyFromHostAsync(float[] src, long srcOffset, long dstOffset, long count, GpuStream stream);

    void copyToHost(float[] dst, long srcOffset, long dstOffset, long count);

    void copyToHostAsync(float[] dst, long srcOffset, long dstOffset, long count, GpuStream stream);

    void copyFromDevice(GpuBuffer src, long srcOff, long dstOff, long bytes, GpuStream stream);

    void memset(byte val);

    void close();
}
