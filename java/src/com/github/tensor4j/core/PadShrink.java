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

/** Shared PAD helpers for eager conv and lazy movement ops. */
public final class PadShrink {

    private PadShrink() {
    }

    public static int[] paddedShape(int[] shape, int[] padArg) {
        int[] out = shape.clone();
        for (int axis = 0; axis < shape.length; axis++) {
            out[axis] += padArg[axis * 2] + padArg[axis * 2 + 1];
        }
        return out;
    }

    public static Tensor applyPad(Tensor input, int[] padArg) {
        int[] inShape = input.shape().dims();
        int[] outShape = paddedShape(inShape, padArg);
        float[] out = new float[Strides.numel(outShape)];
        int rank = inShape.length;
        int[] outIndex = new int[rank];
        int n = input.numel();
        for (int flat = 0; flat < n; flat++) {
            int[] inIndex = Strides.unravel(flat, inShape);
            for (int axis = 0; axis < rank; axis++) {
                outIndex[axis] = inIndex[axis] + padArg[axis * 2];
            }
            out[Strides.ravel(outIndex, outShape)] = input.getFlat(flat);
        }
        return Tensor.of(out, outShape);
    }
}
