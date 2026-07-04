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

import com.github.tensor4j.core.NormShape;
import com.github.tensor4j.core.Tensor;

/** Decomposed group norm (backward via existing lazy rules). */
final class LazyGroupNorm {

    private LazyGroupNorm() {
    }

    static LazyUOp groupNorm(LazyUOp input, LazyUOp weight, LazyUOp bias, int numGroups, int channelAxis, float eps) {
        int[] shape = LazyUOpShapes.infer(input);
        int[] collapsed = NormShape.groupCollapsedShape(shape, channelAxis, numGroups);
        LazyUOp flatGroup = LazyUOp.unary(LazyUOp.Kind.RESHAPE, input, collapsed);
        LazyUOp mean = meanOverAxis(flatGroup, 2, collapsed[2]);
        LazyUOp meanBroadcast = LazyUOp.unary(LazyUOp.Kind.EXPAND, mean, collapsed);
        LazyUOp centered = LazyUOp.binary(LazyUOp.Kind.SUB, flatGroup, meanBroadcast);
        LazyUOp sq = LazyUOp.binary(LazyUOp.Kind.MUL, centered, centered);
        LazyUOp var = meanOverAxis(sq, 2, collapsed[2]);
        LazyUOp epsNode = LazyUOp.buffer(Tensor.of(eps));
        LazyUOp varEps = LazyUOp.binary(LazyUOp.Kind.ADD, var, epsNode);
        LazyUOp invStd = LazyUOp.unary(LazyUOp.Kind.RECIP, LazyUOp.unary(LazyUOp.Kind.SQRT, varEps, null), null);
        LazyUOp invStdBroadcast = LazyUOp.unary(LazyUOp.Kind.EXPAND, invStd, collapsed);
        LazyUOp normalized = LazyUOp.binary(LazyUOp.Kind.MUL, centered, invStdBroadcast);
        LazyUOp normalizedFull = LazyUOp.unary(LazyUOp.Kind.RESHAPE, normalized, shape);
        int channels = shape[NormShape.normalizeAxis(channelAxis, shape.length)];
        LazyUOp weightExpanded = expandChannelVector(weight, shape, channelAxis, channels);
        LazyUOp biasExpanded = expandChannelVector(bias, shape, channelAxis, channels);
        LazyUOp scaled = LazyUOp.binary(LazyUOp.Kind.MUL, normalizedFull, weightExpanded);
        return LazyUOp.binary(LazyUOp.Kind.ADD, scaled, biasExpanded);
    }

    private static LazyUOp meanOverAxis(LazyUOp input, int axis, int count) {
        LazyUOp sumKeep = LazyUOp.unary(LazyUOp.Kind.SUM_AXIS, input, LazySoftmax.axisArg(axis, true));
        LazyUOp countNode = LazyUOp.buffer(Tensor.of((float) count));
        return LazyUOp.binary(LazyUOp.Kind.DIV, sumKeep, countNode);
    }

    private static LazyUOp expandChannelVector(LazyUOp vector, int[] inputShape, int channelAxis, int channels) {
        int[] broadcastShape = NormShape.channelBroadcastShape(inputShape, channelAxis, channels);
        LazyUOp reshaped = LazyUOp.unary(LazyUOp.Kind.RESHAPE, vector, broadcastShape);
        return LazyUOp.unary(LazyUOp.Kind.EXPAND, reshaped, inputShape);
    }
}
