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

/** tinygrad {@code _align_left} + {@code _broadcast_shape} for elementwise ops. */
final class BroadcastShape {

    private BroadcastShape() {
    }

    static int[] compute(int[] left, int[] right) {
        int maxRank = Math.max(left.length, right.length);
        int[] alignedLeft = alignLeft(left, maxRank);
        int[] alignedRight = alignLeft(right, maxRank);
        int[] result = new int[maxRank];
        for (int axis = 0; axis < maxRank; axis++) {
            int leftDim = alignedLeft[axis];
            int rightDim = alignedRight[axis];
            if (leftDim == rightDim) {
                result[axis] = leftDim;
            } else if (leftDim == 1) {
                result[axis] = rightDim;
            } else if (rightDim == 1) {
                result[axis] = leftDim;
            } else {
                throw new IllegalArgumentException("shape mismatch: cannot broadcast "
                        + Arrays.toString(left) + " with " + Arrays.toString(right));
            }
        }
        return result;
    }

    private static int[] alignLeft(int[] shape, int targetRank) {
        if (shape.length > targetRank) {
            throw new IllegalArgumentException("cannot align shape to lower rank");
        }
        int[] aligned = new int[targetRank];
        int pad = targetRank - shape.length;
        Arrays.fill(aligned, 0, pad, 1);
        System.arraycopy(shape, 0, aligned, pad, shape.length);
        return aligned;
    }
}
