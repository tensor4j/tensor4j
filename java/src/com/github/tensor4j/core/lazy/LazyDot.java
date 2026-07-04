/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.core.lazy;

import java.util.Arrays;

/**
 * Lazy dot / matmul via broadcast mul + sum (tinygrad {@code Tensor.dot}).
 *
 * <pre>{@code
 * x = x.reshape(..., 1, K)
 * w = w.reshape(..., 1, K, N).transpose(-1, -2)
 * return (x * w).sum(-1)
 * }</pre>
 */
final class LazyDot {

    private LazyDot() {
    }

    static LazyUOp dot(LazyUOp left, LazyUOp right) {
        int[] xShape = LazyUOpShapes.infer(left);
        int[] wShape = LazyUOpShapes.infer(right);
        int dx = xShape.length;
        int dw = wShape.length;
        if (dx < 1 || dw < 1) {
            throw new IllegalArgumentException("dot requires rank >= 1 tensors");
        }
        int innerX = xShape[dx - 1];
        int wAxis = Math.max(0, dw - 2);
        int innerW = wShape[wAxis];
        if (innerX != innerW) {
            throw new IllegalArgumentException("inner dimensions must match for dot: "
                    + Arrays.toString(xShape) + " vs " + Arrays.toString(wShape));
        }
        int broadcastRank = Math.min(Math.min(dx - 1, dw - 1), 1);

        int[] xReshape = concat(prefix(xShape, dx - 1), ones(broadcastRank), new int[] {innerX});
        LazyUOp x = LazyUOp.unary(LazyUOp.Kind.RESHAPE, left, xReshape);

        int wPrefixLen = Math.max(0, dw - 2);
        int[] wReshape = concat(prefix(wShape, wPrefixLen), ones(broadcastRank), suffix(wShape, wPrefixLen));
        LazyUOp w = LazyUOp.unary(LazyUOp.Kind.RESHAPE, right, wReshape);
        if (dw >= 2) {
            w = LazyUOp.unary(LazyUOp.Kind.PERMUTE, w, swapLastTwo(wReshape.length));
        }

        LazyUOp product = LazyUOp.binary(LazyUOp.Kind.MUL, x, w);
        return LazyUOp.unary(LazyUOp.Kind.SUM_AXIS, product, new int[] {xReshape.length - 1});
    }

    private static int[] prefix(int[] shape, int length) {
        return Arrays.copyOf(shape, length);
    }

    private static int[] suffix(int[] shape, int start) {
        return Arrays.copyOfRange(shape, start, shape.length);
    }

    private static int[] ones(int count) {
        int[] dims = new int[count];
        Arrays.fill(dims, 1);
        return dims;
    }

    private static int[] concat(int[]... parts) {
        int total = 0;
        for (int[] part : parts) {
            total += part.length;
        }
        int[] out = new int[total];
        int offset = 0;
        for (int[] part : parts) {
            System.arraycopy(part, 0, out, offset, part.length);
            offset += part.length;
        }
        return out;
    }

    private static int[] swapLastTwo(int rank) {
        int[] order = new int[rank];
        for (int axis = 0; axis < rank; axis++) {
            order[axis] = axis;
        }
        order[rank - 1] = rank - 2;
        order[rank - 2] = rank - 1;
        return order;
    }
}
