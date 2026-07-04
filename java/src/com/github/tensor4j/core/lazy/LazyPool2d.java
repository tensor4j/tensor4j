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

import com.github.tensor4j.core.Pool2dArg;
import com.github.tensor4j.core.Tensor;

/** Lazy pool2d via tinygrad-style {@code pad + _pool + max/mean}. */
final class LazyPool2d {

    private LazyPool2d() {
    }

    static LazyUOp pool2d(LazyUOp input, int[] arg) {
        Pool2dArg cfg = Pool2dArg.parse(arg);
        int[] inShape = LazyUOpShapes.infer(input);
        int[] outShape = cfg.outputShape(inShape);
        int n = inShape[0];
        int c = inShape[1];
        int oh = outShape[2];
        int ow = outShape[3];
        int kh = cfg.kernelH;
        int kw = cfg.kernelW;
        LazyUOp padded = LazyUOp.unary(LazyUOp.Kind.PAD, input, cfg.padArg(inShape));
        LazyUOp windows = LazyUOp.unary(LazyUOp.Kind.IM2COL, padded, cfg.windowArg(outShape));
        LazyUOp reshaped = LazyUOp.unary(LazyUOp.Kind.RESHAPE, windows,
                new int[] {n, c, oh, ow, kh, kw});
        if (cfg.isMax()) {
            LazyUOp maxKw = LazyUOp.unary(LazyUOp.Kind.MAX_AXIS, reshaped, LazySoftmax.axisArg(5, false));
            LazyUOp maxKh = LazyUOp.unary(LazyUOp.Kind.MAX_AXIS, maxKw, LazySoftmax.axisArg(4, false));
            return maxKh;
        }
        LazyUOp sumKw = LazyUOp.unary(LazyUOp.Kind.SUM_AXIS, reshaped, LazySoftmax.axisArg(5, false));
        LazyUOp sumKh = LazyUOp.unary(LazyUOp.Kind.SUM_AXIS, sumKw, LazySoftmax.axisArg(4, false));
        float scale = 1f / (kh * kw);
        LazyUOp scaleNode = LazyUOp.buffer(Tensor.of(scale));
        return LazyUOp.binary(LazyUOp.Kind.MUL, sumKh, scaleNode);
    }
}
