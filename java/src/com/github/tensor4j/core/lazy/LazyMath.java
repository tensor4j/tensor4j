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

import com.github.tensor4j.core.SoftmaxMath;
import com.github.tensor4j.core.Tensor;

/** Float math constants for tinygrad-style unary ops. */
final class LazyMath {

    static final float LN2 = (float) Math.log(2.0);
    static final float INV_LN2 = 1f / LN2;

    private LazyMath() {
    }

    static int floatArg(float value) {
        return Float.floatToIntBits(value);
    }

    static float floatArg(int bits) {
        return Float.intBitsToFloat(bits);
    }

    static float[] powInto(float[] input, float exponent, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) Math.pow(input[i], exponent);
        }
        return out;
    }

    static float[] log2Into(float[] input, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) (Math.log(input[i]) / Math.log(2.0));
        }
        return out;
    }

    static float[] expInto(float[] input, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) Math.exp(input[i]);
        }
        return out;
    }

    static float[] maxAxisInto(float[] input, int[] shape, int axis, boolean keepdim) {
        return SoftmaxMath.maxAxisInto(input, shape, axis, keepdim);
    }

    static float[] exp2Into(float[] input, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) Math.pow(2.0, input[i]);
        }
        return out;
    }

    static float[] sqrtInto(float[] input, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) Math.sqrt(input[i]);
        }
        return out;
    }

    static float[] maxInto(float[] left, float[] right, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = Math.max(left[i], right[i]);
        }
        return out;
    }

    static float[] gtMaskInto(float[] left, float[] right, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = left[i] > right[i] ? 1f : 0f;
        }
        return out;
    }

    static float[] eqMaskInto(float[] left, float[] right, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = left[i] == right[i] ? 1f : 0f;
        }
        return out;
    }

    static float[] whereInto(float[] cond, float[] ifTrue, float[] ifFalse, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = cond[i] != 0f ? ifTrue[i] : ifFalse[i];
        }
        return out;
    }

    static Tensor contiguousCopy(Tensor input) {
        return Tensor.of(input.toFlatArray(), input.shape().dims());
    }
}
