/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.core;

import java.util.Arrays;

/** Row-major stride utilities (tinygrad flat buffer + shape/strides model). */
public final class Strides {

    private Strides() {
    }

    public static int numel(int[] shape) {
        int total = 1;
        for (int dim : shape) {
            total *= dim;
        }
        return total;
    }

    public static int[] rowMajor(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int axis = shape.length - 1; axis >= 0; axis--) {
            strides[axis] = stride;
            stride *= shape[axis];
        }
        return strides;
    }

    public static boolean isContiguous(int[] shape, int[] strides) {
        if (Arrays.equals(strides, rowMajor(shape))) {
            return true;
        }
        for (int stride : strides) {
            if (stride == 0) {
                return false;
            }
        }
        return Arrays.equals(strides, rowMajor(shape));
    }

    public static int[] alignLeft(int[] shape, int targetRank) {
        if (shape.length > targetRank) {
            throw new IllegalArgumentException("cannot align shape to lower rank");
        }
        int[] aligned = new int[targetRank];
        int pad = targetRank - shape.length;
        Arrays.fill(aligned, 0, pad, 1);
        System.arraycopy(shape, 0, aligned, pad, shape.length);
        return aligned;
    }

    public static int[] inversePermute(int[] order) {
        int[] inverse = new int[order.length];
        for (int axis = 0; axis < order.length; axis++) {
            inverse[order[axis]] = axis;
        }
        return inverse;
    }

    public static float sum(float[] storage, int baseOffset, int[] shape, int[] strides) {
        float total = 0f;
        int n = numel(shape);
        for (int i = 0; i < n; i++) {
            total += read(storage, baseOffset, shape, strides, i);
        }
        return total;
    }

    public static int offset(int baseOffset, int[] indices, int[] strides) {
        int offset = baseOffset;
        for (int axis = 0; axis < indices.length; axis++) {
            offset += indices[axis] * strides[axis];
        }
        return offset;
    }

    public static int flatOffset(int baseOffset, int flatIndex, int[] shape, int[] strides) {
        return offset(baseOffset, unravel(flatIndex, shape), strides);
    }

    public static int ravel(int[] indices, int[] shape) {
        int flat = 0;
        int stride = 1;
        for (int axis = shape.length - 1; axis >= 0; axis--) {
            flat += indices[axis] * stride;
            stride *= shape[axis];
        }
        return flat;
    }

    public static int[] unravel(int flatIndex, int[] shape) {
        int[] indices = new int[shape.length];
        int remaining = flatIndex;
        for (int axis = shape.length - 1; axis >= 0; axis--) {
            indices[axis] = remaining % shape[axis];
            remaining /= shape[axis];
        }
        return indices;
    }

    public static float read(float[] storage, int baseOffset, int[] shape, int[] strides, int flatIndex) {
        return storage[flatOffset(baseOffset, flatIndex, shape, strides)];
    }

    public static void write(float[] storage, int baseOffset, int[] shape, int[] strides, int flatIndex, float value) {
        storage[flatOffset(baseOffset, flatIndex, shape, strides)] = value;
    }
}
