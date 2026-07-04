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

import java.util.Random;

/**
 * Dropout (tinygrad teaching subset): {@code output = input * mask / (1 - p)} during training.
 * {@code mask} values are 0 (drop) or 1 (keep). Use {@link #sampleMask} or {@code dropout(p)} on {@link Tensor}.
 */
public final class DropoutMath {

    private DropoutMath() {
    }

    public static int[] packArg(float p) {
        return new int[] {Float.floatToIntBits(p)};
    }

    public static float pFromArg(int[] arg) {
        return Float.intBitsToFloat(arg[0]);
    }

    public static float scale(float p) {
        if (p < 0f || p >= 1f) {
            throw new IllegalArgumentException("dropout p must be in [0, 1)");
        }
        return 1f / (1f - p);
    }

    public static Tensor sampleMask(int[] shape, float p) {
        return sampleMask(shape, p, TensorRng.threadLocal());
    }

    public static Tensor sampleMask(int[] shape, float p, long seed) {
        return sampleMask(shape, p, TensorRng.seeded(seed));
    }

    public static Tensor sampleMask(int[] shape, float p, Random rng) {
        validateP(p);
        int n = Strides.numel(shape);
        float[] mask = new float[n];
        for (int i = 0; i < n; i++) {
            mask[i] = rng.nextFloat() >= p ? 1f : 0f;
        }
        return Tensor.of(mask, shape);
    }

    public static Tensor forward(Tensor input, Tensor mask, float p) {
        validateMask(input, mask);
        float scale = scale(p);
        float[] in = input.toFlatArray();
        float[] m = mask.toFlatArray();
        float[] out = new float[in.length];
        for (int i = 0; i < in.length; i++) {
            out[i] = in[i] * m[i] * scale;
        }
        return Tensor.of(out, input.shape().dims());
    }

    public static Tensor gradInput(Tensor gradOutput, Tensor mask, float p) {
        validateMask(gradOutput, mask);
        float scale = scale(p);
        float[] gradOut = gradOutput.toFlatArray();
        float[] m = mask.toFlatArray();
        float[] gradIn = new float[gradOut.length];
        for (int i = 0; i < gradOut.length; i++) {
            gradIn[i] = gradOut[i] * m[i] * scale;
        }
        return Tensor.of(gradIn, gradOutput.shape().dims());
    }

    private static void validateP(float p) {
        if (p < 0f || p >= 1f) {
            throw new IllegalArgumentException("dropout p must be in [0, 1)");
        }
    }

    private static void validateMask(Tensor input, Tensor mask) {
        if (!java.util.Arrays.equals(input.shape().dims(), mask.shape().dims())) {
            throw new IllegalArgumentException("dropout mask must match input shape");
        }
    }
}
