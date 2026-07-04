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

/**
 * Layer norm (tinygrad teaching subset): normalize over trailing {@code normalized_shape}
 * per leading slice. {@code weight}/{@code bias} match {@code normalized_shape}.
 */
public final class LayerNormMath {

    public static final class Cache {
        public final float[] mean;
        public final float[] var;
        public final float[] xhat;

        Cache(float[] mean, float[] var, float[] xhat) {
            this.mean = mean;
            this.var = var;
            this.xhat = xhat;
        }
    }

    public static final class ForwardResult {
        public final Tensor output;
        public final Cache cache;

        ForwardResult(Tensor output, Cache cache) {
            this.output = output;
            this.cache = cache;
        }
    }

    private LayerNormMath() {
    }

    public static ForwardResult forward(Tensor input, Tensor weight, Tensor bias, int[] normalizedShape, float eps) {
        int[] shape = input.shape().dims();
        NormShape.validateNormalizedShape(shape, normalizedShape);
        validateAffine(weight, bias, normalizedShape);
        int slices = NormShape.numSlices(shape, normalizedShape);
        int normSize = NormShape.normSize(normalizedShape);
        float[] in = input.toFlatArray();
        float[] gamma = weight.toFlatArray();
        float[] beta = bias.toFlatArray();
        float[] mean = new float[slices];
        float[] var = new float[slices];
        computeStats(in, shape, normalizedShape, mean, var);
        float[] xhat = new float[in.length];
        float[] out = new float[in.length];
        for (int index = 0; index < in.length; index++) {
            int slice = NormShape.sliceOf(index, shape, normalizedShape);
            float invStd = 1f / (float) Math.sqrt(var[slice] + eps);
            xhat[index] = (in[index] - mean[slice]) * invStd;
            out[index] = xhat[index] * gamma[NormShape.affineIndex(index, normalizedShape)]
                    + beta[NormShape.affineIndex(index, normalizedShape)];
        }
        return new ForwardResult(Tensor.of(out, shape), new Cache(mean, var, xhat));
    }

    public static Tensor gradInput(Tensor gradOutput, Tensor weight, int[] normalizedShape, float eps, Cache cache,
            int[] inputShape) {
        int normSize = NormShape.normSize(normalizedShape);
        int slices = NormShape.numSlices(inputShape, normalizedShape);
        float[] gamma = weight.toFlatArray();
        float[] gradOut = gradOutput.toFlatArray();
        float[] gradIn = new float[gradOut.length];
        float[] gradXhat = new float[gradOut.length];
        for (int index = 0; index < gradOut.length; index++) {
            gradXhat[index] = gradOut[index] * gamma[NormShape.affineIndex(index, normalizedShape)];
        }
        for (int slice = 0; slice < slices; slice++) {
            float invStd = 1f / (float) Math.sqrt(cache.var[slice] + eps);
            float sumDy = 0f;
            float sumDyXhat = 0f;
            for (int index = 0; index < gradOut.length; index++) {
                if (NormShape.sliceOf(index, inputShape, normalizedShape) != slice) {
                    continue;
                }
                sumDy += gradXhat[index];
                sumDyXhat += gradXhat[index] * cache.xhat[index];
            }
            for (int index = 0; index < gradOut.length; index++) {
                if (NormShape.sliceOf(index, inputShape, normalizedShape) != slice) {
                    continue;
                }
                gradIn[index] = invStd * (gradXhat[index] - sumDy / normSize
                        - cache.xhat[index] * sumDyXhat / normSize);
            }
        }
        return Tensor.of(gradIn, inputShape);
    }

    public static Tensor gradWeight(Tensor gradOutput, Cache cache, int[] normalizedShape) {
        float[] gradOut = gradOutput.toFlatArray();
        float[] gradW = new float[NormShape.normSize(normalizedShape)];
        for (int index = 0; index < gradOut.length; index++) {
            gradW[NormShape.affineIndex(index, normalizedShape)] += gradOut[index] * cache.xhat[index];
        }
        return Tensor.of(gradW, normalizedShape);
    }

    public static Tensor gradBias(Tensor gradOutput, int[] normalizedShape) {
        float[] gradOut = gradOutput.toFlatArray();
        float[] gradB = new float[NormShape.normSize(normalizedShape)];
        for (int index = 0; index < gradOut.length; index++) {
            gradB[NormShape.affineIndex(index, normalizedShape)] += gradOut[index];
        }
        return Tensor.of(gradB, normalizedShape);
    }

    private static void computeStats(float[] in, int[] shape, int[] normalizedShape, float[] mean, float[] var) {
        int slices = NormShape.numSlices(shape, normalizedShape);
        int normSize = NormShape.normSize(normalizedShape);
        float[] sum = new float[slices];
        float[] sumSq = new float[slices];
        for (int index = 0; index < in.length; index++) {
            int slice = NormShape.sliceOf(index, shape, normalizedShape);
            sum[slice] += in[index];
            sumSq[slice] += in[index] * in[index];
        }
        for (int slice = 0; slice < slices; slice++) {
            mean[slice] = sum[slice] / normSize;
            var[slice] = sumSq[slice] / normSize - mean[slice] * mean[slice];
        }
    }

    private static void validateAffine(Tensor weight, Tensor bias, int[] normalizedShape) {
        if (!java.util.Arrays.equals(weight.shape().dims(), normalizedShape)
                || !java.util.Arrays.equals(bias.shape().dims(), normalizedShape)) {
            throw new IllegalArgumentException("weight/bias must match normalized_shape");
        }
    }
}
