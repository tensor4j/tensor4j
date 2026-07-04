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

/** Shape/strides metadata for interpreting a {@link StorageBuffer} (tinygrad movement-op view). */
public final class TensorLayout {

    private final int[] shape;
    private final int[] strides;

    public TensorLayout(int[] shape, int[] strides) {
        this.shape = shape.clone();
        this.strides = strides.clone();
    }

    public static TensorLayout contiguous(int[] shape) {
        return new TensorLayout(shape, Strides.rowMajor(shape));
    }

    public int[] shape() {
        return shape.clone();
    }

    public int[] strides() {
        return strides.clone();
    }

    public int rank() {
        return shape.length;
    }

    public int dim(int axis) {
        return shape[axis];
    }

    public int numel() {
        return Strides.numel(shape);
    }

    public boolean isContiguous() {
        return Strides.isContiguous(shape, strides);
    }

    public TensorLayout reshape(int[] newShape) {
        if (Strides.numel(newShape) != numel()) {
            throw new IllegalArgumentException("reshape incompatible element counts");
        }
        if (!isContiguous()) {
            throw new IllegalArgumentException("reshape view requires contiguous layout");
        }
        return new TensorLayout(newShape, Strides.rowMajor(newShape));
    }

    public TensorLayout permute(int[] order) {
        if (order.length != rank()) {
            throw new IllegalArgumentException("permute order rank mismatch");
        }
        int[] newShape = new int[order.length];
        int[] newStrides = new int[order.length];
        for (int axis = 0; axis < order.length; axis++) {
            int source = order[axis];
            newShape[axis] = shape[source];
            newStrides[axis] = strides[source];
        }
        return new TensorLayout(newShape, newStrides);
    }

    public TensorLayout transpose2d() {
        if (rank() != 2) {
            throw new IllegalArgumentException("transpose2d requires rank-2 tensor");
        }
        return permute(new int[] {1, 0});
    }

    /** Broadcast view (tinygrad {@code expand} / stride-0 axes). */
    public TensorLayout expand(int[] targetShape) {
        if (targetShape.length < rank()) {
            throw new IllegalArgumentException("expand target rank must be >= source rank");
        }
        int pad = targetShape.length - rank();
        int[] alignedShape = Strides.alignLeft(shape, targetShape.length);
        int[] alignedStrides = new int[targetShape.length];
        for (int axis = 0; axis < targetShape.length; axis++) {
            int sourceDim = alignedShape[axis];
            int targetDim = targetShape[axis];
            if (sourceDim == targetDim) {
                alignedStrides[axis] = axis < pad ? 0 : strides[axis - pad];
            } else if (sourceDim == 1) {
                alignedStrides[axis] = 0;
            } else {
                throw new IllegalArgumentException("cannot expand " + Arrays.toString(shape) + " to " + Arrays.toString(targetShape));
            }
        }
        return new TensorLayout(targetShape, alignedStrides);
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof TensorLayout layout)) {
            return false;
        }
        return Arrays.equals(shape, layout.shape) && Arrays.equals(strides, layout.strides);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(shape) ^ Arrays.hashCode(strides);
    }

    @Override
    public String toString() {
        return "shape=" + Arrays.toString(shape) + " strides=" + Arrays.toString(strides);
    }
}
