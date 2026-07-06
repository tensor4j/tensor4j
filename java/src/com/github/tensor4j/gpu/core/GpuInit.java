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

import java.util.Random;

/**
 * Weight initialization strategies for GPU neural network parameters.
 */
public final class GpuInit {

    private GpuInit() {
    }

    public static float[] uniform(int numel, float scale, Random rng) {
        float[] w = new float[numel];
        for (int i = 0; i < numel; i++) {
            w[i] = (rng.nextFloat() - 0.5f) * 2f * scale;
        }
        return w;
    }

    public static float[] xavierUniform(int fanIn, int fanOut, Random rng) {
        float scale = (float) Math.sqrt(6.0 / (fanIn + fanOut));
        return uniform(fanIn * fanOut, scale, rng);
    }

    public static float[] heUniform(int fanIn, int fanOut, Random rng) {
        float scale = (float) Math.sqrt(2.0 / fanIn);
        return uniform(fanIn * fanOut, scale, rng);
    }

    public static float[] zeros(int numel) {
        return new float[numel];
    }

    public static float[] ones(int numel) {
        float[] w = new float[numel];
        java.util.Arrays.fill(w, 1f);
        return w;
    }

    public static GpuAutogradTensor createParam(GpuDevice device, float[] data, int... shape) {
        return GpuAutogradTensor.fromHost(device, data, shape).requiresGrad(true);
    }
}
