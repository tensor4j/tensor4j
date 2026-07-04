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
 * NCHW conv2d dispatch (tinygrad {@code conv2d}: im2col default, Winograd for 3x3 s1).
 */
public final class Conv2dMath {

    /** @deprecated use {@link Conv2dArg#ARG_LEN} */
    public static final int ARG_LEN = Conv2dArg.ARG_LEN;

    private Conv2dMath() {
    }

    public static int[] defaultArg() {
        return Conv2dArg.defaultPacked();
    }

    public static int[] arg(int stride, int padBeforeH, int padAfterH, int padBeforeW, int padAfterW, int groups) {
        return Conv2dArg.packed(stride, padBeforeH, padAfterH, padBeforeW, padAfterW, groups);
    }

    public static int[] arg(int stride, int padBeforeH, int padAfterH, int padBeforeW, int padAfterW, int groups,
            int dilation) {
        return Conv2dArg.packed(stride, padBeforeH, padAfterH, padBeforeW, padAfterW, groups, dilation);
    }

    public static int[] outputShape(int[] inputNchw, int[] weightOihw, int[] arg) {
        return Conv2dArg.parse(arg).outputShape(inputNchw, weightOihw);
    }

    public static int[] depthwiseArg(int[] arg, int channels) {
        int[] packed = arg.length == Conv2dArg.ARG_LEN ? arg.clone() : Conv2dArg.parse(arg).packed();
        packed[6] = channels;
        return packed;
    }

    public static Tensor forward(Tensor input, Tensor weight, int[] arg) {
        Conv2dArg cfg = Conv2dArg.parse(arg);
        if (cfg.canWinograd(weight.shape().dims())) {
            return ConvWinograd.forward(input, weight, cfg);
        }
        return ConvIm2Col.forward(input, weight, cfg);
    }

    public static Tensor forwardIm2Col(Tensor input, Tensor weight, int[] arg) {
        return ConvIm2Col.forward(input, weight, Conv2dArg.parse(arg));
    }

    public static Tensor forwardWinograd(Tensor input, Tensor weight, int[] arg) {
        return ConvWinograd.forward(input, weight, Conv2dArg.parse(arg));
    }

    public static Tensor gradInput(Tensor gradOutput, Tensor weight, int[] arg, int[] inputShape) {
        Conv2dArg cfg = Conv2dArg.parse(arg);
        int[] wShape = weight.shape().dims();
        cfg.validateShapes(inputShape, wShape);
        float[] gradIn = new float[Strides.numel(inputShape)];
        float[] gradOut = gradOutput.toFlatArray();
        float[] wFlat = weight.toFlatArray();
        int n = inputShape[0];
        int inC = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outC = wShape[0];
        int kH = wShape[2];
        int kW = wShape[3];
        int groups = cfg.groups;
        int cinPerGroup = inC / groups;
        int coutPerGroup = outC / groups;
        int[] outShape = cfg.outputShape(inputShape, wShape);
        int outH = outShape[2];
        int outW = outShape[3];
        for (int batch = 0; batch < n; batch++) {
            for (int ic = 0; ic < inC; ic++) {
                int group = ic / cinPerGroup;
                for (int ih = 0; ih < inH; ih++) {
                    for (int iw = 0; iw < inW; iw++) {
                        float sum = 0f;
                        for (int oc = group * coutPerGroup; oc < (group + 1) * coutPerGroup; oc++) {
                            int icLocal = ic - group * cinPerGroup;
                            for (int kh = 0; kh < kH; kh++) {
                                for (int kw = 0; kw < kW; kw++) {
                                    int oh = ih + cfg.padBeforeH - kh * cfg.dilH;
                                    int ow = iw + cfg.padBeforeW - kw * cfg.dilW;
                                    if (oh % cfg.strideH == 0 && ow % cfg.strideW == 0) {
                                        oh /= cfg.strideH;
                                        ow /= cfg.strideW;
                                        if (oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                                            int goIndex = ConvIm2Col.indexNchw(batch, oc, oh, ow, outC, outH, outW);
                                            int wIndex = ConvIm2Col.indexOihw(oc, icLocal, kh, kw, outC, cinPerGroup,
                                                    kH, kW);
                                            sum += gradOut[goIndex] * wFlat[wIndex];
                                        }
                                    }
                                }
                            }
                        }
                        gradIn[ConvIm2Col.indexNchw(batch, ic, ih, iw, inC, inH, inW)] = sum;
                    }
                }
            }
        }
        return Tensor.of(gradIn, inputShape);
    }

    public static Tensor gradWeight(Tensor gradOutput, Tensor input, int[] arg, int[] weightShape) {
        Conv2dArg cfg = Conv2dArg.parse(arg);
        int[] inShape = input.shape().dims();
        cfg.validateShapes(inShape, weightShape);
        float[] gradW = new float[Strides.numel(weightShape)];
        float[] gradOut = gradOutput.toFlatArray();
        float[] inFlat = input.toFlatArray();
        int n = inShape[0];
        int inC = inShape[1];
        int inH = inShape[2];
        int inW = inShape[3];
        int outC = weightShape[0];
        int kH = weightShape[2];
        int kW = weightShape[3];
        int groups = cfg.groups;
        int cinPerGroup = inC / groups;
        int coutPerGroup = outC / groups;
        int[] outShape = cfg.outputShape(inShape, weightShape);
        int outH = outShape[2];
        int outW = outShape[3];
        for (int oc = 0; oc < outC; oc++) {
            int group = oc / coutPerGroup;
            for (int ic = 0; ic < cinPerGroup; ic++) {
                int inChannel = group * cinPerGroup + ic;
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        float sum = 0f;
                        for (int batch = 0; batch < n; batch++) {
                            for (int oh = 0; oh < outH; oh++) {
                                for (int ow = 0; ow < outW; ow++) {
                                    int ih = oh * cfg.strideH + kh * cfg.dilH - cfg.padBeforeH;
                                    int iw = ow * cfg.strideW + kw * cfg.dilW - cfg.padBeforeW;
                                    if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                        int inIndex = ConvIm2Col.indexNchw(batch, inChannel, ih, iw, inC, inH, inW);
                                        int goIndex = ConvIm2Col.indexNchw(batch, oc, oh, ow, outC, outH, outW);
                                        sum += inFlat[inIndex] * gradOut[goIndex];
                                    }
                                }
                            }
                        }
                        gradW[ConvIm2Col.indexOihw(oc, ic, kh, kw, outC, cinPerGroup, kH, kW)] = sum;
                    }
                }
            }
        }
        return Tensor.of(gradW, weightShape);
    }
}
