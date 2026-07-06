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

import java.util.Arrays;

/**
 * GPU-resident float32 tensor: shape/strides metadata + device buffer.
 * Operations dispatch GPU kernels; fall back to CPU if no device available.
 */
public final class GpuTensor implements AutoCloseable {

    private final int[] shape;
    private final int[] strides;
    private final GpuBuffer buffer;
    private final int offset;

    public GpuTensor(GpuBuffer buffer, int[] shape, int[] strides, int offset) {
        this.buffer = buffer;
        this.shape = shape.clone();
        this.strides = strides.clone();
        this.offset = offset;
    }

    public static GpuTensor fromHost(GpuDevice device, float[] data, int... shape) {
        int numel = 1;
        for (int d : shape) {
            numel *= d;
        }
        if (numel != data.length) {
            throw new IllegalArgumentException("data length " + data.length + " != shape " + Arrays.toString(shape));
        }
        GpuBuffer buf = device.allocate((long) numel * 4L);
        buf.copyFromHost(data, 0, 0, numel);
        int[] strides = computeStrides(shape);
        return new GpuTensor(buf, shape, strides, 0);
    }

    public float[] toHost() {
        float[] result = new float[numel()];
        buffer.copyToHost(result, 0, 0, numel());
        return result;
    }

    public int[] shape() {
        return shape.clone();
    }

    public int[] strides() {
        return strides.clone();
    }

    public int numel() {
        int n = 1;
        for (int d : shape) {
            n *= d;
        }
        return n;
    }

    public int ndim() {
        return shape.length;
    }

    public GpuBuffer buffer() {
        return buffer;
    }

    public int offset() {
        return offset;
    }

    public GpuTensor reshape(int... newShape) {
        int newNumel = 1;
        for (int d : newShape) {
            newNumel *= d;
        }
        if (newNumel != numel()) {
            throw new IllegalArgumentException(
                "cannot reshape tensor of " + numel() + " elements into " + java.util.Arrays.toString(newShape));
        }
        int[] newStrides = computeStrides(newShape);
        return new GpuTensor(buffer, newShape, newStrides, offset);
    }

    public GpuTensor squeeze() {
        int count = 0;
        for (int d : shape) {
            if (d != 1) count++;
        }
        if (count == shape.length) return this;
        int[] newShape = new int[count];
        int idx = 0;
        for (int d : shape) {
            if (d != 1) newShape[idx++] = d;
        }
        int[] newStrides = computeStrides(newShape);
        return new GpuTensor(buffer, newShape, newStrides, offset);
    }

    public GpuTensor unsqueeze(int dim) {
        int newLen = shape.length + 1;
        if (dim < 0) dim = newLen + dim;
        if (dim < 0 || dim >= newLen) {
            throw new IllegalArgumentException("invalid dim " + dim + " for unsqueeze");
        }
        int[] newShape = new int[newLen];
        int[] newStrides = new int[newLen];
        int si = 0;
        for (int i = 0; i < newLen; i++) {
            if (i == dim) {
                newShape[i] = 1;
                newStrides[i] = 1;
            } else {
                newShape[i] = shape[si];
                newStrides[i] = strides[si];
                si++;
            }
        }
        return new GpuTensor(buffer, newShape, newStrides, offset);
    }

    @Override
    public void close() {
        buffer.close();
    }

    static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }
}
