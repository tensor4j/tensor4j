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
 * Group norm (tinygrad teaching subset): normalize over {@code (C/G, *trailing)} per leading slice and group.
 * {@code weight}/{@code bias} length {@code C} at {@code channelAxis}.
 */
public final class GroupNormMath {

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

    private GroupNormMath() {
    }

    public static int[] packArg(int numGroups, int channelAxis, float eps) {
        return new int[] {numGroups, channelAxis, Float.floatToIntBits(eps)};
    }

    public static int numGroupsFromArg(int[] arg) {
        return arg[0];
    }

    public static int channelAxisFromArg(int[] arg) {
        return arg[1];
    }

    public static float epsFromArg(int[] arg) {
        return Float.intBitsToFloat(arg[2]);
    }

    public static ForwardResult forward(Tensor input, Tensor weight, Tensor bias, int numGroups, int channelAxis,
            float eps) {
        int[] shape = input.shape().dims();
        channelAxis = NormShape.normalizeAxis(channelAxis, shape.length);
        int channels = shape[channelAxis];
        validateVectors(weight, bias, channels);
        int[] collapsed = NormShape.groupCollapsedShape(shape, channelAxis, numGroups);
        int normSize = collapsed[2];
        int groupCount = collapsed[0] * collapsed[1];
        float[] in = input.toFlatArray();
        float[] gamma = weight.toFlatArray();
        float[] beta = bias.toFlatArray();
        float[] mean = new float[groupCount];
        float[] var = new float[groupCount];
        computeStats(in, shape, channelAxis, numGroups, mean, var);
        float[] xhat = new float[in.length];
        float[] out = new float[in.length];
        for (int index = 0; index < in.length; index++) {
            int key = NormShape.groupKey(index, shape, channelAxis, numGroups);
            float invStd = 1f / (float) Math.sqrt(var[key] + eps);
            xhat[index] = (in[index] - mean[key]) * invStd;
            int c = NormShape.channelIndex(index, shape, channelAxis);
            out[index] = xhat[index] * gamma[c] + beta[c];
        }
        return new ForwardResult(Tensor.of(out, shape), new Cache(mean, var, xhat));
    }

    public static Tensor gradInput(Tensor gradOutput, Tensor weight, int numGroups, int channelAxis, float eps,
            Cache cache, int[] inputShape) {
        channelAxis = NormShape.normalizeAxis(channelAxis, inputShape.length);
        int[] collapsed = NormShape.groupCollapsedShape(inputShape, channelAxis, numGroups);
        int normSize = collapsed[2];
        int groupCount = collapsed[0] * collapsed[1];
        float[] gamma = weight.toFlatArray();
        float[] gradOut = gradOutput.toFlatArray();
        float[] gradIn = new float[gradOut.length];
        float[] gradXhat = new float[gradOut.length];
        for (int index = 0; index < gradOut.length; index++) {
            gradXhat[index] = gradOut[index] * gamma[NormShape.channelIndex(index, inputShape, channelAxis)];
        }
        for (int key = 0; key < groupCount; key++) {
            float invStd = 1f / (float) Math.sqrt(cache.var[key] + eps);
            float sumDy = 0f;
            float sumDyXhat = 0f;
            for (int index = 0; index < gradOut.length; index++) {
                if (NormShape.groupKey(index, inputShape, channelAxis, numGroups) != key) {
                    continue;
                }
                sumDy += gradXhat[index];
                sumDyXhat += gradXhat[index] * cache.xhat[index];
            }
            for (int index = 0; index < gradOut.length; index++) {
                if (NormShape.groupKey(index, inputShape, channelAxis, numGroups) != key) {
                    continue;
                }
                gradIn[index] = invStd * (gradXhat[index] - sumDy / normSize
                        - cache.xhat[index] * sumDyXhat / normSize);
            }
        }
        return Tensor.of(gradIn, inputShape);
    }

    public static Tensor gradWeight(Tensor gradOutput, Cache cache, int[] inputShape, int channelAxis) {
        channelAxis = NormShape.normalizeAxis(channelAxis, inputShape.length);
        int channels = inputShape[channelAxis];
        float[] gradOut = gradOutput.toFlatArray();
        float[] gradW = new float[channels];
        for (int index = 0; index < gradOut.length; index++) {
            gradW[NormShape.channelIndex(index, inputShape, channelAxis)] += gradOut[index] * cache.xhat[index];
        }
        return Tensor.of(gradW, channels);
    }

    public static Tensor gradBias(Tensor gradOutput, int[] inputShape, int channelAxis) {
        channelAxis = NormShape.normalizeAxis(channelAxis, inputShape.length);
        int channels = inputShape[channelAxis];
        float[] gradOut = gradOutput.toFlatArray();
        float[] gradB = new float[channels];
        for (int index = 0; index < gradOut.length; index++) {
            gradB[NormShape.channelIndex(index, inputShape, channelAxis)] += gradOut[index];
        }
        return Tensor.of(gradB, channels);
    }

    private static void computeStats(float[] in, int[] shape, int channelAxis, int numGroups, float[] mean,
            float[] var) {
        int[] collapsed = NormShape.groupCollapsedShape(shape, channelAxis, numGroups);
        int normSize = collapsed[2];
        int groupCount = collapsed[0] * collapsed[1];
        float[] sum = new float[groupCount];
        float[] sumSq = new float[groupCount];
        for (int index = 0; index < in.length; index++) {
            int key = NormShape.groupKey(index, shape, channelAxis, numGroups);
            sum[key] += in[index];
            sumSq[key] += in[index] * in[index];
        }
        for (int key = 0; key < groupCount; key++) {
            mean[key] = sum[key] / normSize;
            var[key] = sumSq[key] / normSize - mean[key] * mean[key];
        }
    }

    private static void validateVectors(Tensor weight, Tensor bias, int channels) {
        if (weight.numel() != channels || bias.numel() != channels) {
            throw new IllegalArgumentException("weight/bias length must match channels");
        }
    }
}
