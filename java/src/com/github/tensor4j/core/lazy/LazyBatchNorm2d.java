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

import com.github.tensor4j.core.Tensor;

/**
 * NCHW batch norm (tinygrad {@code Tensor.batchnorm} decomposition).
 * Eval uses supplied stats; train computes batch mean/var then applies the same subgraph.
 */
final class LazyBatchNorm2d {

    private LazyBatchNorm2d() {
    }

    /** Decomposed eval graph (backward via existing lazy rules). */
    static LazyUOp batchNorm2dEval(LazyUOp input, LazyUOp weight, LazyUOp bias, LazyUOp mean, LazyUOp var,
            float eps) {
        int[] shape = LazyUOpShapes.infer(input);
        LazyUOp mean4 = expandChannelVector(mean, shape);
        LazyUOp var4 = expandChannelVector(var, shape);
        LazyUOp weight4 = expandChannelVector(weight, shape);
        LazyUOp bias4 = expandChannelVector(bias, shape);
        LazyUOp centered = LazyUOp.binary(LazyUOp.Kind.SUB, input, mean4);
        LazyUOp normalized = normalize(centered, var4, eps, shape);
        LazyUOp scaled = LazyUOp.binary(LazyUOp.Kind.MUL, normalized, weight4);
        return LazyUOp.binary(LazyUOp.Kind.ADD, scaled, bias4);
    }

    /**
     * Decomposed training graph (tinygrad {@code var_mean} + {@code batchnorm}); backward via existing lazy rules.
     */
    static LazyUOp batchNorm2dTrain(LazyUOp input, LazyUOp weight, LazyUOp bias, float eps) {
        int[] shape = LazyUOpShapes.infer(input);
        int channels = shape[1];
        int reduceSize = shape[0] * shape[2] * shape[3];
        int[] channelFlatShape = new int[] {channels, reduceSize};
        LazyUOp flat = channelFlatFromNchw(input, shape, channelFlatShape);
        LazyUOp mean = meanOverAxis(flat, 1, reduceSize);
        LazyUOp meanBroadcast = LazyUOp.unary(LazyUOp.Kind.EXPAND, mean, channelFlatShape);
        LazyUOp centered = LazyUOp.binary(LazyUOp.Kind.SUB, flat, meanBroadcast);
        LazyUOp sq = LazyUOp.binary(LazyUOp.Kind.MUL, centered, centered);
        LazyUOp var = meanOverAxis(sq, 1, reduceSize);
        LazyUOp normalizedFlat = normalizeFromVariance(centered, var, eps, channelFlatShape);
        LazyUOp normalized = nchwFromChannelFlat(normalizedFlat, shape);
        LazyUOp weight4 = expandChannelVector(weight, shape);
        LazyUOp bias4 = expandChannelVector(bias, shape);
        LazyUOp scaled = LazyUOp.binary(LazyUOp.Kind.MUL, normalized, weight4);
        return LazyUOp.binary(LazyUOp.Kind.ADD, scaled, bias4);
    }

    private static LazyUOp normalize(LazyUOp centered, LazyUOp var, float eps, int[] nchw) {
        LazyUOp var4 = var;
        if (LazyUOpShapes.infer(var).length == 1) {
            var4 = expandChannelVector(var, nchw);
        }
        return normalizeFromVariance(centered, var4, eps, nchw);
    }

    private static LazyUOp normalizeFromVariance(LazyUOp centered, LazyUOp var, float eps, int[] shape) {
        LazyUOp epsNode = LazyUOp.buffer(Tensor.of(eps));
        LazyUOp varEps = LazyUOp.binary(LazyUOp.Kind.ADD, var, epsNode);
        LazyUOp invStd = LazyUOp.unary(LazyUOp.Kind.RECIP, LazyUOp.unary(LazyUOp.Kind.SQRT, varEps, null), null);
        LazyUOp invStdBroadcast = LazyUOp.unary(LazyUOp.Kind.EXPAND, invStd, shape);
        return LazyUOp.binary(LazyUOp.Kind.MUL, centered, invStdBroadcast);
    }

    private static LazyUOp channelFlatFromNchw(LazyUOp input, int[] nchw, int[] channelFlatShape) {
        LazyUOp chFirst = LazyUOp.unary(LazyUOp.Kind.PERMUTE, input, new int[] {1, 0, 2, 3});
        return LazyUOp.unary(LazyUOp.Kind.RESHAPE, chFirst, channelFlatShape);
    }

    private static LazyUOp nchwFromChannelFlat(LazyUOp channelFlat, int[] nchw) {
        LazyUOp chFirst = LazyUOp.unary(LazyUOp.Kind.RESHAPE, channelFlat, new int[] {nchw[1], nchw[0], nchw[2], nchw[3]});
        return LazyUOp.unary(LazyUOp.Kind.PERMUTE, chFirst, new int[] {1, 0, 2, 3});
    }

    private static LazyUOp meanOverAxis(LazyUOp input, int axis, int count) {
        LazyUOp sumKeep = LazyUOp.unary(LazyUOp.Kind.SUM_AXIS, input, LazySoftmax.axisArg(axis, true));
        LazyUOp countNode = LazyUOp.buffer(Tensor.of((float) count));
        return LazyUOp.binary(LazyUOp.Kind.DIV, sumKeep, countNode);
    }

    private static LazyUOp expandChannelVector(LazyUOp vector, int[] nchw) {
        int channels = nchw[1];
        LazyUOp reshaped = LazyUOp.unary(LazyUOp.Kind.RESHAPE, vector, new int[] {1, channels, 1, 1});
        return LazyUOp.unary(LazyUOp.Kind.EXPAND, reshaped, nchw);
    }
}
