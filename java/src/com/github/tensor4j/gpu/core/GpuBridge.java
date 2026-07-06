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

import com.github.tensor4j.core.Tensor;

/**
 * Bridge between CPU-side {@link Tensor} (with autograd) and GPU-side {@link GpuTensor}.
 * Enables compute on GPU while preserving CPU autograd graph.
 */
public final class GpuBridge {

    private final GpuDevice device;
    private final GpuTensorMath math;

    public GpuBridge() {
        this(GpuRuntime.defaultDevice());
    }

    public GpuBridge(GpuDevice device) {
        this.device = device;
        this.math = new GpuTensorMath(device);
    }

    public GpuTensor toGpu(Tensor t) {
        int[] shape = t.shape().dims();
        return GpuTensor.fromHost(device, t.data(), shape);
    }

    public Tensor toTensor(GpuTensor t) {
        float[] data = t.toHost();
        int[] shape = t.shape();
        return Tensor.of(data, shape);
    }

    public Tensor add(Tensor a, Tensor b) {
        try (GpuTensor ga = toGpu(a); GpuTensor gb = toGpu(b)) {
            GpuTensor gc = math.add(ga, gb);
            return toTensor(gc);
        }
    }

    public Tensor sub(Tensor a, Tensor b) {
        try (GpuTensor ga = toGpu(a); GpuTensor gb = toGpu(b)) {
            GpuTensor gc = math.sub(ga, gb);
            return toTensor(gc);
        }
    }

    public Tensor mul(Tensor a, Tensor b) {
        try (GpuTensor ga = toGpu(a); GpuTensor gb = toGpu(b)) {
            GpuTensor gc = math.mul(ga, gb);
            return toTensor(gc);
        }
    }

    public Tensor div(Tensor a, Tensor b) {
        try (GpuTensor ga = toGpu(a); GpuTensor gb = toGpu(b)) {
            GpuTensor gc = math.div(ga, gb);
            return toTensor(gc);
        }
    }

    public Tensor relu(Tensor a) {
        try (GpuTensor ga = toGpu(a)) {
            GpuTensor gc = math.relu(ga);
            return toTensor(gc);
        }
    }

    public Tensor neg(Tensor a) {
        try (GpuTensor ga = toGpu(a)) {
            GpuTensor gc = math.neg(ga);
            return toTensor(gc);
        }
    }

    public Tensor scale(Tensor a, float s) {
        try (GpuTensor ga = toGpu(a)) {
            GpuTensor gc = math.scale(ga, s);
            return toTensor(gc);
        }
    }

    public Tensor exp(Tensor a) {
        try (GpuTensor ga = toGpu(a)) {
            GpuTensor gc = math.exp(ga);
            return toTensor(gc);
        }
    }

    public Tensor log(Tensor a) {
        try (GpuTensor ga = toGpu(a)) {
            GpuTensor gc = math.log(ga);
            return toTensor(gc);
        }
    }

    public Tensor matmul(Tensor a, Tensor b) {
        try (GpuTensor ga = toGpu(a); GpuTensor gb = toGpu(b)) {
            GpuTensor gc = math.matmul(ga, gb);
            return toTensor(gc);
        }
    }

    public Tensor sqrt(Tensor a) {
        try (GpuTensor ga = toGpu(a)) {
            GpuTensor gc = math.sqrt(ga);
            return toTensor(gc);
        }
    }

    public Tensor sigmoid(Tensor a) {
        try (GpuTensor ga = toGpu(a)) {
            GpuTensor gc = math.sigmoid(ga);
            return toTensor(gc);
        }
    }

    public Tensor tanh(Tensor a) {
        try (GpuTensor ga = toGpu(a)) {
            GpuTensor gc = math.tanh(ga);
            return toTensor(gc);
        }
    }

    public Tensor squaredDifference(Tensor a, Tensor b) {
        try (GpuTensor ga = toGpu(a); GpuTensor gb = toGpu(b)) {
            GpuTensor gc = math.squaredDifference(ga, gb);
            return toTensor(gc);
        }
    }

    public Tensor sum(Tensor a) {
        try (GpuTensor ga = toGpu(a)) {
            GpuTensor gc = math.sum(ga);
            return toTensor(gc);
        }
    }

    public Tensor softmax(Tensor a) {
        try (GpuTensor ga = toGpu(a)) {
            GpuTensor gc = math.softmax(ga);
            return toTensor(gc);
        }
    }

    public GpuDevice device() {
        return device;
    }
}
