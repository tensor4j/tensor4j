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

import java.util.ArrayList;
import java.util.List;
import com.github.tensor4j.gpu.core.*;

/**
 * CPU reference device — runs all operations on the host.
 * Used for CI testing and as correctness oracle.
 */
public final class CpuDevice implements GpuDevice {

    private final String name;
    private boolean closed;

    public CpuDevice() {
        this.name = "CPU Reference (" + Runtime.getRuntime().availableProcessors() + " cores)";
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public int computeCapabilityMajor() {
        return 0;
    }

    @Override
    public int computeCapabilityMinor() {
        return 0;
    }

    @Override
    public long totalMemoryBytes() {
        return Runtime.getRuntime().maxMemory();
    }

    @Override
    public long freeMemoryBytes() {
        return Runtime.getRuntime().freeMemory();
    }

    @Override
    public GpuStream createStream() {
        return new CpuStream();
    }

    @Override
    public GpuBuffer allocate(long bytes) {
        return new CpuBuffer(bytes);
    }

    @Override
    public GpuBuffer allocateManaged(long bytes) {
        return allocate(bytes);
    }

    @Override
    public GpuProgram compile(String source, String kernelName) {
        return new CpuProgram(kernelName);
    }

    @Override
    public void synchronize() {
    }

    @Override
    public void close() {
        closed = true;
    }

    public boolean isClosed() {
        return closed;
    }
}
