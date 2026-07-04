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
 * NCHW conv_transpose2d via upsample + conv2d (tinygrad {@code conv_transpose2d} teaching subset).
 * Weight layout {@code [Cin, Cout/groups, kH, kW]}.
 */
public final class ConvTranspose2dMath {

    private ConvTranspose2dMath() {
    }

    public static int[] defaultArg() {
        return ConvTranspose2dArg.defaultPacked();
    }

    public static int[] outputShape(int[] inputNchw, int[] weightIohw, int[] arg) {
        return ConvTranspose2dArg.parse(arg).outputShape(inputNchw, weightIohw);
    }

    public static Tensor forward(Tensor input, Tensor weight, int[] arg) {
        ConvTranspose2dArg cfg = ConvTranspose2dArg.parse(arg);
        int[] inShape = input.shape().dims();
        int[] wShape = weight.shape().dims();
        cfg.validateShapes(inShape, wShape);
        Tensor upsampled = upsampleInput(input, cfg);
        Tensor convWeight = prepareConvWeight(weight, cfg);
        int[] convArg = conv2dArg(cfg, wShape);
        return Conv2dMath.forward(upsampled, convWeight, convArg);
    }

    public static Tensor gradInput(Tensor gradOutput, Tensor weight, int[] arg, int[] inputShape) {
        ConvTranspose2dArg cfg = ConvTranspose2dArg.parse(arg);
        int[] wShape = weight.shape().dims();
        int[] upShape = upsampledShape(inputShape, cfg);
        Tensor convWeight = prepareConvWeight(weight, cfg);
        int[] convArg = conv2dArg(cfg, wShape);
        Tensor upGrad = Conv2dMath.gradInput(gradOutput, convWeight, convArg, upShape);
        return downsampleGrad(upGrad, cfg, inputShape);
    }

    public static Tensor gradWeight(Tensor gradOutput, Tensor input, int[] arg, int[] weightShape) {
        ConvTranspose2dArg cfg = ConvTranspose2dArg.parse(arg);
        Tensor upsampled = upsampleInput(input, cfg);
        int[] convWShape = new int[] {
                weightShape[1] * cfg.groups, weightShape[0] / cfg.groups, weightShape[2], weightShape[3]
        };
        int[] convArg = conv2dArg(cfg, weightShape);
        Tensor gradConvWeight = Conv2dMath.gradWeight(gradOutput, upsampled, convArg, convWShape);
        return unprepareConvWeightGrad(gradConvWeight, weightShape, cfg);
    }

    static int[] upsampledShape(int[] inputNchw, ConvTranspose2dArg cfg) {
        if (cfg.strideH == 1 && cfg.strideW == 1) {
            return inputNchw.clone();
        }
        int[] out = inputNchw.clone();
        out[2] = inputNchw[2] * cfg.strideH - (cfg.strideH - 1);
        out[3] = inputNchw[3] * cfg.strideW - (cfg.strideW - 1);
        return out;
    }

    private static Tensor upsampleInput(Tensor input, ConvTranspose2dArg cfg) {
        int[] inShape = input.shape().dims();
        if (cfg.strideH == 1 && cfg.strideW == 1) {
            return input;
        }
        int[] upShape = upsampledShape(inShape, cfg);
        int n = inShape[0];
        int c = inShape[1];
        int inH = inShape[2];
        int inW = inShape[3];
        int upH = upShape[2];
        int upW = upShape[3];
        float[] out = new float[Strides.numel(upShape)];
        float[] inFlat = input.toFlatArray();
        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < c; channel++) {
                for (int ih = 0; ih < inH; ih++) {
                    for (int iw = 0; iw < inW; iw++) {
                        int oh = ih * cfg.strideH;
                        int ow = iw * cfg.strideW;
                        out[ConvIm2Col.indexNchw(batch, channel, oh, ow, c, upH, upW)]
                                = inFlat[ConvIm2Col.indexNchw(batch, channel, ih, iw, c, inH, inW)];
                    }
                }
            }
        }
        return Tensor.of(out, upShape);
    }

    private static Tensor downsampleGrad(Tensor upGrad, ConvTranspose2dArg cfg, int[] inputShape) {
        if (cfg.strideH == 1 && cfg.strideW == 1) {
            return upGrad;
        }
        int n = inputShape[0];
        int c = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int[] upShape = upsampledShape(inputShape, cfg);
        int upH = upShape[2];
        int upW = upShape[3];
        float[] gradIn = new float[Strides.numel(inputShape)];
        float[] upFlat = upGrad.toFlatArray();
        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < c; channel++) {
                for (int ih = 0; ih < inH; ih++) {
                    for (int iw = 0; iw < inW; iw++) {
                        int oh = ih * cfg.strideH;
                        int ow = iw * cfg.strideW;
                        gradIn[ConvIm2Col.indexNchw(batch, channel, ih, iw, c, inH, inW)]
                                = upFlat[ConvIm2Col.indexNchw(batch, channel, oh, ow, c, upH, upW)];
                    }
                }
            }
        }
        return Tensor.of(gradIn, inputShape);
    }

    private static int[] conv2dArg(ConvTranspose2dArg cfg, int[] weightIohw) {
        int kH = weightIohw[2];
        int kW = weightIohw[3];
        int padBeforeH = (kH - 1) - cfg.padBeforeH;
        int padAfterH = (kH - 1) - cfg.padAfterH + cfg.outPadH;
        int padBeforeW = (kW - 1) - cfg.padBeforeW;
        int padAfterW = (kW - 1) - cfg.padAfterW + cfg.outPadW;
        return Conv2dArg.packed(1, padBeforeH, padAfterH, padBeforeW, padAfterW, cfg.groups);
    }

    private static Tensor prepareConvWeight(Tensor weight, ConvTranspose2dArg cfg) {
        int[] wShape = weight.shape().dims();
        int inC = wShape[0];
        int coutPerGroup = wShape[1];
        int outC = coutPerGroup * cfg.groups;
        int cinPerGroup = inC / cfg.groups;
        int kH = wShape[2];
        int kW = wShape[3];
        float[] src = weight.toFlatArray();
        float[] dst = new float[src.length];
        for (int ic = 0; ic < inC; ic++) {
            int group = ic / cinPerGroup;
            for (int ocLocal = 0; ocLocal < coutPerGroup; ocLocal++) {
                int oc = group * coutPerGroup + ocLocal;
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        float value = src[indexIohw(ic, ocLocal, kh, kw, inC, coutPerGroup, kH, kW)];
                        dst[ConvIm2Col.indexOihw(oc, ic - group * cinPerGroup, kH - 1 - kh, kW - 1 - kw,
                                outC, cinPerGroup, kH, kW)] = value;
                    }
                }
            }
        }
        return Tensor.of(dst, new int[] {outC, cinPerGroup, kH, kW});
    }

    private static Tensor unprepareConvWeightGrad(Tensor gradConvWeight, int[] weightIohw, ConvTranspose2dArg cfg) {
        int inC = weightIohw[0];
        int coutPerGroup = weightIohw[1];
        int kH = weightIohw[2];
        int kW = weightIohw[3];
        int cinPerGroup = inC / cfg.groups;
        int outC = coutPerGroup * cfg.groups;
        float[] src = gradConvWeight.toFlatArray();
        float[] dst = new float[Strides.numel(weightIohw)];
        for (int ic = 0; ic < inC; ic++) {
            int group = ic / cinPerGroup;
            for (int ocLocal = 0; ocLocal < coutPerGroup; ocLocal++) {
                int oc = group * coutPerGroup + ocLocal;
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        float value = src[ConvIm2Col.indexOihw(oc, ic - group * cinPerGroup, kH - 1 - kh,
                                kW - 1 - kw, outC, cinPerGroup, kH, kW)];
                        dst[indexIohw(ic, ocLocal, kh, kw, inC, coutPerGroup, kH, kW)] = value;
                    }
                }
            }
        }
        return Tensor.of(dst, weightIohw);
    }

    static int indexIohw(int ic, int oc, int kh, int kw, int inC, int outC, int kH, int kW) {
        return ((ic * outC + oc) * kH + kh) * kW + kw;
    }
}
