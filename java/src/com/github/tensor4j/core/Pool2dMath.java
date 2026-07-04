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
 * NCHW pool2d (tinygrad {@code _pool + max/mean} teaching subset).
 */
public final class Pool2dMath {

    public static final class ForwardResult {
        public final Tensor output;
        /** Window-local argmax {@code kh * kW + kw} for max-pool backward. */
        public final int[] argmax;
        /** Unpadded flat spatial index {@code ih * W + iw} for max_unpool2d. */
        public final int[] flatSpatialIndex;

        ForwardResult(Tensor output, int[] argmax, int[] flatSpatialIndex) {
            this.output = output;
            this.argmax = argmax;
            this.flatSpatialIndex = flatSpatialIndex;
        }
    }

    private Pool2dMath() {
    }

    public static int[] defaultMaxArg() {
        return Pool2dArg.maxPacked(2, 2);
    }

    public static int[] outputShape(int[] inputNchw, int[] arg) {
        return Pool2dArg.parse(arg).outputShape(inputNchw);
    }

    public static Tensor forward(Tensor input, int[] arg) {
        return forwardWithMeta(input, arg).output;
    }

    public static MaxPool2dResult maxPoolWithIndices(Tensor input, int[] arg) {
        ForwardResult result = forwardWithMeta(input, arg);
        float[] idx = new float[result.flatSpatialIndex.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = result.flatSpatialIndex[i];
        }
        return new MaxPool2dResult(result.output, Tensor.of(idx, result.output.shape().dims()));
    }

    public static ForwardResult forwardWithMeta(Tensor input, int[] arg) {
        Pool2dArg cfg = Pool2dArg.parse(arg);
        int[] inShape = input.shape().dims();
        Tensor padded = PadShrink.applyPad(input, cfg.padArg(inShape));
        int[] outShape = cfg.outputShape(inShape);
        int[] padShape = padded.shape().dims();
        int n = inShape[0];
        int channels = inShape[1];
        int inH = padShape[2];
        int inW = padShape[3];
        int outH = outShape[2];
        int outW = outShape[3];
        int kH = cfg.kernelH;
        int kW = cfg.kernelW;
        float[] out = new float[Strides.numel(outShape)];
        int[] argmax = cfg.isMax() ? new int[out.length] : null;
        int[] flatSpatial = cfg.isMax() ? new int[out.length] : null;
        int srcH = inShape[2];
        int srcW = inShape[3];
        float poolSize = kH * kW;
        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < channels; channel++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        float acc = cfg.isMax() ? Float.NEGATIVE_INFINITY : 0f;
                        int bestKh = 0;
                        int bestKw = 0;
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                int ih = oh * cfg.strideH + kh;
                                int iw = ow * cfg.strideW + kw;
                                float value = 0f;
                                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                    value = padded.getFlat(ConvIm2Col.indexNchw(batch, channel, ih, iw,
                                            channels, inH, inW));
                                }
                                if (cfg.isMax()) {
                                    if (value > acc) {
                                        acc = value;
                                        bestKh = kh;
                                        bestKw = kw;
                                    }
                                } else {
                                    acc += value;
                                }
                            }
                        }
                        if (!cfg.isMax()) {
                            acc /= poolSize;
                        }
                        int outIndex = ConvIm2Col.indexNchw(batch, channel, oh, ow, channels, outH, outW);
                        out[outIndex] = acc;
                        if (argmax != null) {
                            int ihOrig = oh * cfg.strideH + bestKh - cfg.padBeforeH;
                            int iwOrig = ow * cfg.strideW + bestKw - cfg.padBeforeW;
                            argmax[outIndex] = bestKh * kW + bestKw;
                            flatSpatial[outIndex] = ihOrig * srcW + iwOrig;
                        }
                    }
                }
            }
        }
        return new ForwardResult(Tensor.of(out, outShape), argmax, flatSpatial);
    }

    public static Tensor gradInput(Tensor gradOutput, int[] arg, int[] inputShape, int[] argmax) {
        Pool2dArg cfg = Pool2dArg.parse(arg);
        int[] outShape = cfg.outputShape(inputShape);
        int n = inputShape[0];
        int channels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outH = outShape[2];
        int outW = outShape[3];
        int kH = cfg.kernelH;
        int kW = cfg.kernelW;
        float[] gradIn = new float[Strides.numel(inputShape)];
        float[] gradOut = gradOutput.toFlatArray();
        float poolSize = kH * kW;
        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < channels; channel++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        int outIndex = ConvIm2Col.indexNchw(batch, channel, oh, ow, channels, outH, outW);
                        float upstream = gradOut[outIndex];
                        if (cfg.isMax()) {
                            int flat = argmax[outIndex];
                            int kh = flat / kW;
                            int kw = flat % kW;
                            int ih = oh * cfg.strideH + kh - cfg.padBeforeH;
                            int iw = ow * cfg.strideW + kw - cfg.padBeforeW;
                            if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                gradIn[ConvIm2Col.indexNchw(batch, channel, ih, iw, channels, inH, inW)] += upstream;
                            }
                        } else {
                            float share = upstream / poolSize;
                            for (int kh = 0; kh < kH; kh++) {
                                for (int kw = 0; kw < kW; kw++) {
                                    int ih = oh * cfg.strideH + kh - cfg.padBeforeH;
                                    int iw = ow * cfg.strideW + kw - cfg.padBeforeW;
                                    if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                        gradIn[ConvIm2Col.indexNchw(batch, channel, ih, iw, channels, inH, inW)]
                                                += share;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return Tensor.of(gradIn, inputShape);
    }
}
