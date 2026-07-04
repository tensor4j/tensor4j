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
 * NCHW batch norm (tinygrad {@code batchnorm} teaching subset).
 * {@code weight}/{@code bias}/{@code mean}/{@code var} are per-channel vectors length {@code C}.
 */
public final class BatchNorm2dMath {

    public static final class TrainCache {
        public final float[] mean;
        public final float[] var;
        public final float[] xhat;

        TrainCache(float[] mean, float[] var, float[] xhat) {
            this.mean = mean;
            this.var = var;
            this.xhat = xhat;
        }
    }

    private BatchNorm2dMath() {
    }

    public static int[] packArg(float eps) {
        return new int[] {Float.floatToIntBits(eps)};
    }

    public static float epsFromArg(int[] arg) {
        return Float.intBitsToFloat(arg[0]);
    }

    public static Tensor forwardEval(Tensor input, Tensor weight, Tensor bias, Tensor mean, Tensor var, float eps) {
        int[] shape = input.shape().dims();
        validateVectors(weight, bias, mean, var, shape[1]);
        float[] out = new float[Strides.numel(shape)];
        float[] in = input.toFlatArray();
        float[] gamma = weight.toFlatArray();
        float[] beta = bias.toFlatArray();
        float[] mu = mean.toFlatArray();
        float[] sigma2 = var.toFlatArray();
        forwardEvalInto(in, out, shape, gamma, beta, mu, sigma2, eps);
        return Tensor.of(out, shape);
    }

    public static class TrainForward {
        public final Tensor output;
        public final TrainCache cache;

        TrainForward(Tensor output, TrainCache cache) {
            this.output = output;
            this.cache = cache;
        }
    }

    public static TrainForward forwardTrain(Tensor input, Tensor weight, Tensor bias, float eps) {
        int[] shape = input.shape().dims();
        int channels = shape[1];
        validateVectors(weight, bias, channels);
        float[] in = input.toFlatArray();
        float[] gamma = weight.toFlatArray();
        float[] beta = bias.toFlatArray();
        float[] mean = new float[channels];
        float[] var = new float[channels];
        computeBatchStats(in, shape, mean, var);
        float[] xhat = new float[in.length];
        float[] out = new float[in.length];
        for (int index = 0; index < in.length; index++) {
            int c = channelOf(index, shape);
            float invStd = 1f / (float) Math.sqrt(var[c] + eps);
            xhat[index] = (in[index] - mean[c]) * invStd;
            out[index] = xhat[index] * gamma[c] + beta[c];
        }
        return new TrainForward(Tensor.of(out, shape), new TrainCache(mean, var, xhat));
    }

    public static Tensor gradInputTrain(Tensor gradOutput, Tensor weight, float eps, TrainCache cache, int[] inputShape) {
        float[] gamma = weight.toFlatArray();
        float[] gradOut = gradOutput.toFlatArray();
        float[] gradIn = new float[gradOut.length];
        int channels = inputShape[1];
        int reduce = inputShape[0] * inputShape[2] * inputShape[3];
        for (int c = 0; c < channels; c++) {
            float invStd = 1f / (float) Math.sqrt(cache.var[c] + eps);
            float sumDy = 0f;
            float sumDyXhat = 0f;
            for (int index = 0; index < gradOut.length; index++) {
                if (channelOf(index, inputShape) != c) {
                    continue;
                }
                float dy = gradOut[index] * gamma[c];
                sumDy += dy;
                sumDyXhat += dy * cache.xhat[index];
            }
            for (int index = 0; index < gradOut.length; index++) {
                if (channelOf(index, inputShape) != c) {
                    continue;
                }
                float dy = gradOut[index] * gamma[c];
                gradIn[index] = invStd * (dy - sumDy / reduce - cache.xhat[index] * sumDyXhat / reduce);
            }
        }
        return Tensor.of(gradIn, inputShape);
    }

    public static Tensor gradInputEval(Tensor gradOutput, Tensor weight, Tensor var, float eps, int[] inputShape) {
        int channels = inputShape[1];
        float[] gamma = weight.toFlatArray();
        float[] sigma2 = var.toFlatArray();
        float[] gradOut = gradOutput.toFlatArray();
        float[] gradIn = new float[gradOut.length];
        for (int index = 0; index < gradOut.length; index++) {
            int c = channelOf(index, inputShape);
            float invStd = 1f / (float) Math.sqrt(sigma2[c] + eps);
            gradIn[index] = gradOut[index] * gamma[c] * invStd;
        }
        return Tensor.of(gradIn, inputShape);
    }

    public static Tensor gradWeight(Tensor gradOutput, TrainCache cache, int[] inputShape) {
        int channels = inputShape[1];
        float[] gradOut = gradOutput.toFlatArray();
        float[] gradW = new float[channels];
        for (int index = 0; index < gradOut.length; index++) {
            gradW[channelOf(index, inputShape)] += gradOut[index] * cache.xhat[index];
        }
        return Tensor.of(gradW, channels);
    }

    public static Tensor gradBias(Tensor gradOutput, int[] inputShape) {
        int channels = inputShape[1];
        float[] gradOut = gradOutput.toFlatArray();
        float[] gradB = new float[channels];
        for (int index = 0; index < gradOut.length; index++) {
            gradB[channelOf(index, inputShape)] += gradOut[index];
        }
        return Tensor.of(gradB, channels);
    }

    private static void forwardEvalInto(float[] in, float[] out, int[] shape, float[] gamma, float[] beta,
            float[] mu, float[] sigma2, float eps) {
        for (int index = 0; index < in.length; index++) {
            int c = channelOf(index, shape);
            float xhat = (in[index] - mu[c]) / (float) Math.sqrt(sigma2[c] + eps);
            out[index] = xhat * gamma[c] + beta[c];
        }
    }

    private static void computeBatchStats(float[] in, int[] shape, float[] mean, float[] var) {
        int channels = shape[1];
        int reduce = shape[0] * shape[2] * shape[3];
        float[] sum = new float[channels];
        float[] sumSq = new float[channels];
        for (int index = 0; index < in.length; index++) {
            int c = channelOf(index, shape);
            sum[c] += in[index];
            sumSq[c] += in[index] * in[index];
        }
        for (int c = 0; c < channels; c++) {
            mean[c] = sum[c] / reduce;
            var[c] = sumSq[c] / reduce - mean[c] * mean[c];
        }
    }

    private static int channelOf(int flat, int[] nchw) {
        int spatial = nchw[2] * nchw[3];
        return (flat / spatial) % nchw[1];
    }

    private static void validateVectors(Tensor weight, Tensor bias, Tensor mean, Tensor var, int channels) {
        validateVectors(weight, bias, channels);
        if (mean.numel() != channels || var.numel() != channels) {
            throw new IllegalArgumentException("mean/var length must match channels");
        }
    }

    private static void validateVectors(Tensor weight, Tensor bias, int channels) {
        if (weight.numel() != channels || bias.numel() != channels) {
            throw new IllegalArgumentException("weight/bias length must match channels");
        }
    }
}
