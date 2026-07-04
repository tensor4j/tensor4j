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

/** Decomposed layer norm over trailing {@code normalized_shape} (backward via existing lazy rules). */
final class LazyLayerNorm {

    private LazyLayerNorm() {
    }

    static LazyUOp layerNorm(LazyUOp input, LazyUOp weight, LazyUOp bias, int[] normalizedShape, float eps) {
        int[] shape = LazyUOpShapes.infer(input);
        int[] collapsed = NormShape.collapsedShape(shape, normalizedShape);
        LazyUOp flat = LazyUOp.unary(LazyUOp.Kind.RESHAPE, input, collapsed);
        LazyUOp mean = meanOverAxis(flat, 1, collapsed[1]);
        LazyUOp meanBroadcast = LazyUOp.unary(LazyUOp.Kind.EXPAND, mean, collapsed);
        LazyUOp centered = LazyUOp.binary(LazyUOp.Kind.SUB, flat, meanBroadcast);
        LazyUOp sq = LazyUOp.binary(LazyUOp.Kind.MUL, centered, centered);
        LazyUOp var = meanOverAxis(sq, 1, collapsed[1]);
        LazyUOp epsNode = LazyUOp.buffer(Tensor.of(eps));
        LazyUOp varEps = LazyUOp.binary(LazyUOp.Kind.ADD, var, epsNode);
        LazyUOp invStd = LazyUOp.unary(LazyUOp.Kind.RECIP, LazyUOp.unary(LazyUOp.Kind.SQRT, varEps, null), null);
        LazyUOp invStdBroadcast = LazyUOp.unary(LazyUOp.Kind.EXPAND, invStd, collapsed);
        LazyUOp normalized = LazyUOp.binary(LazyUOp.Kind.MUL, centered, invStdBroadcast);
        LazyUOp normalizedFull = LazyUOp.unary(LazyUOp.Kind.RESHAPE, normalized, shape);
        LazyUOp weightExpanded = expandAffine(weight, shape, normalizedShape);
        LazyUOp biasExpanded = expandAffine(bias, shape, normalizedShape);
        LazyUOp scaled = LazyUOp.binary(LazyUOp.Kind.MUL, normalizedFull, weightExpanded);
        return LazyUOp.binary(LazyUOp.Kind.ADD, scaled, biasExpanded);
    }

    private static LazyUOp meanOverAxis(LazyUOp input, int axis, int count) {
        LazyUOp sumKeep = LazyUOp.unary(LazyUOp.Kind.SUM_AXIS, input, LazySoftmax.axisArg(axis, true));
        LazyUOp countNode = LazyUOp.buffer(Tensor.of((float) count));
        return LazyUOp.binary(LazyUOp.Kind.DIV, sumKeep, countNode);
    }

    private static LazyUOp expandAffine(LazyUOp affine, int[] inputShape, int[] normalizedShape) {
        int[] broadcastShape = NormShape.affineBroadcastShape(inputShape, normalizedShape);
        LazyUOp reshaped = LazyUOp.unary(LazyUOp.Kind.RESHAPE, affine, broadcastShape);
        return LazyUOp.unary(LazyUOp.Kind.EXPAND, reshaped, inputShape);
    }
}
