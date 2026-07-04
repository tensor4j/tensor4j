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

import com.github.tensor4j.core.Conv2dArg;
import com.github.tensor4j.core.ConvIm2Col;
import com.github.tensor4j.core.Strides;

/**
 * Lazy conv2d via tinygrad-style {@code pad + im2col + mul + sum} (groups=1) or monolithic {@code CONV2D}.
 */
final class LazyConv2d {

    private LazyConv2d() {
    }

    static LazyUOp conv2d(LazyUOp input, LazyUOp weight, int[] arg) {
        Conv2dArg cfg = Conv2dArg.parse(arg);
        int[] inShape = LazyUOpShapes.infer(input);
        int[] wShape = LazyUOpShapes.infer(weight);
        cfg.outputShape(inShape, wShape);
        if (cfg.groups == 1) {
            return conv2dIm2colGraph(input, weight, cfg, inShape, wShape);
        }
        return LazyUOp.binary(LazyUOp.Kind.CONV2D, input, weight, cfg.packed());
    }

    /** Monolithic conv for backward / grouped fallback. */
    static LazyUOp conv2dMonolithic(LazyUOp input, LazyUOp weight, int[] arg) {
        Conv2dArg cfg = Conv2dArg.parse(arg);
        Conv2dArg.parse(arg).outputShape(LazyUOpShapes.infer(input), LazyUOpShapes.infer(weight));
        return LazyUOp.binary(LazyUOp.Kind.CONV2D, input, weight, cfg.packed());
    }

    private static LazyUOp conv2dIm2colGraph(LazyUOp input, LazyUOp weight, Conv2dArg cfg, int[] inShape,
            int[] wShape) {
        int[] outShape = cfg.outputShape(inShape, wShape);
        int n = inShape[0];
        int oc = wShape[0];
        int ic = wShape[1];
        int kh = wShape[2];
        int kw = wShape[3];
        int oh = outShape[2];
        int ow = outShape[3];
        LazyUOp padded = LazyUOp.unary(LazyUOp.Kind.PAD, input, cfg.padArg(inShape));
        LazyUOp windows = LazyUOp.unary(LazyUOp.Kind.IM2COL, padded, cfg.im2colArg(wShape, outShape));
        LazyUOp win = LazyUOp.unary(LazyUOp.Kind.RESHAPE, windows,
                new int[] {n, 1, oh, ow, ic, kh, kw});
        LazyUOp winExpanded = LazyUOp.unary(LazyUOp.Kind.EXPAND, win, new int[] {n, oc, oh, ow, ic, kh, kw});
        LazyUOp wExpanded = LazyUOp.unary(LazyUOp.Kind.RESHAPE, weight, new int[] {1, oc, 1, 1, ic, kh, kw});
        wExpanded = LazyUOp.unary(LazyUOp.Kind.EXPAND, wExpanded, new int[] {n, oc, oh, ow, ic, kh, kw});
        LazyUOp product = LazyUOp.binary(LazyUOp.Kind.MUL, winExpanded, wExpanded);
        LazyUOp sumKw = LazyUOp.unary(LazyUOp.Kind.SUM_AXIS, product, new int[] {6});
        LazyUOp sumKh = LazyUOp.unary(LazyUOp.Kind.SUM_AXIS, sumKw, new int[] {5});
        LazyUOp sumIc = LazyUOp.unary(LazyUOp.Kind.SUM_AXIS, sumKh, new int[] {4});
        return LazyUOp.unary(LazyUOp.Kind.RESHAPE, sumIc, new int[] {n, oc, oh, ow});
    }

    static int[] im2colWindowShape(int[] paddedNchw, int[] im2colArg) {
        int groups = im2colArg[8];
        int cinPerGroup = paddedNchw[1] / groups;
        return new int[] {
                paddedNchw[0], groups, cinPerGroup, im2colArg[6], im2colArg[7], im2colArg[0], im2colArg[1]
        };
    }

    static float[] realizeIm2Col(com.github.tensor4j.core.Tensor padded, int[] im2colArg) {
        int[] paddedShape = padded.shape().dims();
        int[] weightShape = new int[] {0, paddedShape[1] / im2colArg[8], im2colArg[0], im2colArg[1]};
        Conv2dArg cfg = Conv2dArg.im2colOnPadded(im2colArg[2], im2colArg[3], im2colArg[8], im2colArg[4], im2colArg[5]);
        int[] outShape = new int[] {paddedShape[0], 0, im2colArg[6], im2colArg[7]};
        return ConvIm2Col.extractWindows(padded.toFlatArray(), paddedShape, weightShape, cfg, outShape);
    }

    static int[] im2colGradArg(int[] im2colArg, int[] paddedShape) {
        int[] packed = new int[im2colArg.length + paddedShape.length];
        System.arraycopy(im2colArg, 0, packed, 0, im2colArg.length);
        System.arraycopy(paddedShape, 0, packed, im2colArg.length, paddedShape.length);
        return packed;
    }

    static int[] paddedShapeFromGradArg(int[] gradArg) {
        return java.util.Arrays.copyOfRange(gradArg, 9, gradArg.length);
    }

    static int[] im2colArgFromGradArg(int[] gradArg) {
        return java.util.Arrays.copyOfRange(gradArg, 0, 9);
    }
}
