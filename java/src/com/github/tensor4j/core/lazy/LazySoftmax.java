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

/**
 * Lazy softmax via tinygrad {@code _softmax} decomposition (max/sub/exp/sum/div).
 */
final class LazySoftmax {

    private LazySoftmax() {
    }

    static LazyUOp softmax(LazyUOp input, int axis) {
        int rank = LazyUOpShapes.infer(input).length;
        int normalized = SoftmaxMath.normalizeAxis(axis, rank);
        LazyUOp maxKeep = LazyUOp.unary(LazyUOp.Kind.MAX_AXIS, input, axisArg(normalized, true));
        LazyUOp shifted = LazyUOp.binary(LazyUOp.Kind.SUB, input, maxKeep);
        LazyUOp exp = LazyUOp.unary(LazyUOp.Kind.EXP, shifted, null);
        LazyUOp sumKeep = LazyUOp.unary(LazyUOp.Kind.SUM_AXIS, exp, axisArg(normalized, true));
        return LazyUOp.binary(LazyUOp.Kind.DIV, exp, sumKeep);
    }

    static int[] axisArg(int axis, boolean keepdim) {
        return keepdim ? new int[] {axis, 1} : new int[] {axis};
    }

    static LazyUOp logSoftmax(LazyUOp input, int axis) {
        int rank = LazyUOpShapes.infer(input).length;
        int normalized = SoftmaxMath.normalizeAxis(axis, rank);
        LazyUOp maxKeep = LazyUOp.unary(LazyUOp.Kind.MAX_AXIS, input, axisArg(normalized, true));
        LazyUOp shifted = LazyUOp.binary(LazyUOp.Kind.SUB, input, maxKeep);
        LazyUOp exp = LazyUOp.unary(LazyUOp.Kind.EXP, shifted, null);
        LazyUOp sumKeep = LazyUOp.unary(LazyUOp.Kind.SUM_AXIS, exp, axisArg(normalized, true));
        LazyUOp logSum = LazyUOp.unary(LazyUOp.Kind.LOG2, sumKeep, null);
        LazyUOp logSumNat = LazyUOp.binary(LazyUOp.Kind.MUL, logSum,
                LazyUOp.buffer(Tensor.of(LazyMath.LN2)));
        return LazyUOp.binary(LazyUOp.Kind.SUB, shifted, logSumNat);
    }
}
