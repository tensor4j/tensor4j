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

import com.github.tensor4j.core.PadShrink;
import com.github.tensor4j.core.Strides;
import com.github.tensor4j.core.Tensor;
import java.util.Arrays;

/** PAD / SHRINK helpers (tinygrad movement ops teaching subset). */
final class LazyPadShrink {

    private LazyPadShrink() {
    }

    /** {@code arg}: per-axis {@code [before, after]} padding counts. */
    static int[] paddedShape(int[] shape, int[] padArg) {
        return PadShrink.paddedShape(shape, padArg);
    }

    static int[] shrinkShape(int[] shape, int[] shrinkArg) {
        int[] out = new int[shape.length];
        for (int axis = 0; axis < shape.length; axis++) {
            int start = shrinkArg[axis * 2];
            int end = shrinkArg[axis * 2 + 1];
            out[axis] = end - start;
        }
        return out;
    }

    static Tensor applyPad(Tensor input, int[] padArg) {
        return PadShrink.applyPad(input, padArg);
    }

    static Tensor applyShrink(Tensor input, int[] shrinkArg) {
        int[] inShape = input.shape().dims();
        int[] outShape = shrinkShape(inShape, shrinkArg);
        float[] out = new float[Strides.numel(outShape)];
        int rank = inShape.length;
        for (int flat = 0; flat < out.length; flat++) {
            int[] outIndex = Strides.unravel(flat, outShape);
            int[] inIndex = new int[rank];
            for (int axis = 0; axis < rank; axis++) {
                inIndex[axis] = outIndex[axis] + shrinkArg[axis * 2];
            }
            out[flat] = input.get(inIndex);
        }
        return Tensor.of(out, outShape);
    }

    /** PAD backward: shrink gradient to the pre-pad region. */
    static int[] padBackwardShrinkArg(int[] srcShape, int[] padArg) {
        int[] shrink = new int[srcShape.length * 2];
        for (int axis = 0; axis < srcShape.length; axis++) {
            int before = padArg[axis * 2];
            shrink[axis * 2] = before;
            shrink[axis * 2 + 1] = before + srcShape[axis];
        }
        return shrink;
    }

    static int[] shrinkBackwardPadArg(int[] srcShape, int[] shrinkArg) {
        int[] pad = new int[srcShape.length * 2];
        for (int axis = 0; axis < srcShape.length; axis++) {
            int before = shrinkArg[axis * 2];
            int kept = shrinkArg[axis * 2 + 1] - shrinkArg[axis * 2];
            pad[axis * 2] = before;
            pad[axis * 2 + 1] = srcShape[axis] - kept - before;
        }
        return pad;
    }

    static int[] validatePadArg(int[] shape, int[] padArg) {
        if (padArg.length != shape.length * 2) {
            throw new IllegalArgumentException("pad arg length " + padArg.length
                    + " != 2 * rank " + shape.length);
        }
        return padArg.clone();
    }

    static int[] validateShrinkArg(int[] shape, int[] shrinkArg) {
        if (shrinkArg.length != shape.length * 2) {
            throw new IllegalArgumentException("shrink arg length " + shrinkArg.length
                    + " != 2 * rank " + shape.length);
        }
        for (int axis = 0; axis < shape.length; axis++) {
            int start = shrinkArg[axis * 2];
            int end = shrinkArg[axis * 2 + 1];
            if (start < 0 || end > shape[axis] || start >= end) {
                throw new IllegalArgumentException("invalid shrink " + Arrays.toString(shrinkArg)
                        + " for shape " + Arrays.toString(shape));
            }
        }
        return shrinkArg.clone();
    }
}
