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

import com.github.tensor4j.core.PadShrink;

/**
 * im2col conv2d (tinygrad {@code pad + _pool + mul + sum} teaching subset).
 */
public final class ConvIm2Col {

    private ConvIm2Col() {
    }

    public static Tensor forward(Tensor input, Tensor weight, Conv2dArg cfg) {
        int[] inShape = input.shape().dims();
        int[] wShape = weight.shape().dims();
        cfg.validateShapes(inShape, wShape);
        Tensor padded = PadShrink.applyPad(input, cfg.padArg(inShape));
        int[] outShape = cfg.outputShape(inShape, wShape);
        int[] winShape = windowShape(inShape, wShape, cfg, outShape);
        float[] windows = extractWindows(padded.toFlatArray(), padded.shape().dims(), wShape, cfg, outShape);
        return contract(windows, winShape, weight.toFlatArray(), wShape, cfg, outShape);
    }

    /** Window tensor shape {@code [N, G, Cin/G, OH, OW, kH, kW]}. */
    public static int[] windowShape(int[] inputNchw, int[] weightOihw, Conv2dArg cfg, int[] outputNchw) {
        int groups = cfg.groups;
        return new int[] {
                inputNchw[0], groups, weightOihw[1], outputNchw[2], outputNchw[3], weightOihw[2], weightOihw[3]
        };
    }

    public static float[] extractWindows(float[] padded, int[] paddedShape, int[] weightShape, Conv2dArg cfg,
            int[] outShape) {
        int n = paddedShape[0];
        int inC = paddedShape[1];
        int inH = paddedShape[2];
        int inW = paddedShape[3];
        int groups = cfg.groups;
        int cinPerGroup = inC / groups;
        int kH = weightShape[2];
        int kW = weightShape[3];
        int outH = outShape[2];
        int outW = outShape[3];
        int[] winShape = windowShape(new int[] {n, inC, inH, inW}, weightShape, cfg, outShape);
        float[] windows = new float[Strides.numel(winShape)];
        for (int batch = 0; batch < n; batch++) {
            for (int group = 0; group < groups; group++) {
                for (int ic = 0; ic < cinPerGroup; ic++) {
                    int channel = group * cinPerGroup + ic;
                    for (int oh = 0; oh < outH; oh++) {
                        for (int ow = 0; ow < outW; ow++) {
                            for (int kh = 0; kh < kH; kh++) {
                                for (int kw = 0; kw < kW; kw++) {
                                    int ih = oh * cfg.strideH + kh * cfg.dilH;
                                    int iw = ow * cfg.strideW + kw * cfg.dilW;
                                    float value = 0f;
                                    if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                        value = padded[indexNchw(batch, channel, ih, iw, inC, inH, inW)];
                                    }
                                    windows[indexWindow(batch, group, ic, oh, ow, kh, kw, winShape)] = value;
                                }
                            }
                        }
                    }
                }
            }
        }
        return windows;
    }

    private static Tensor contract(float[] windows, int[] winShape, float[] weight, int[] wShape, Conv2dArg cfg,
            int[] outShape) {
        int n = outShape[0];
        int outC = outShape[1];
        int outH = outShape[2];
        int outW = outShape[3];
        int groups = cfg.groups;
        int cinPerGroup = wShape[1];
        int coutPerGroup = outC / groups;
        int kH = wShape[2];
        int kW = wShape[3];
        float[] out = new float[Strides.numel(outShape)];
        for (int batch = 0; batch < n; batch++) {
            for (int oc = 0; oc < outC; oc++) {
                int group = oc / coutPerGroup;
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        float sum = 0f;
                        for (int ic = 0; ic < cinPerGroup; ic++) {
                            for (int kh = 0; kh < kH; kh++) {
                                for (int kw = 0; kw < kW; kw++) {
                                    float win = windows[indexWindow(batch, group, ic, oh, ow, kh, kw, winShape)];
                                    float w = weight[indexOihw(oc, ic, kh, kw, outC, cinPerGroup, kH, kW)];
                                    sum += win * w;
                                }
                            }
                        }
                        out[indexNchw(batch, oc, oh, ow, outC, outH, outW)] = sum;
                    }
                }
            }
        }
        return Tensor.of(out, outShape);
    }

    static int indexWindow(int n, int g, int ic, int oh, int ow, int kh, int kw, int[] shape) {
        int flat = n;
        flat = flat * shape[1] + g;
        flat = flat * shape[2] + ic;
        flat = flat * shape[3] + oh;
        flat = flat * shape[4] + ow;
        flat = flat * shape[5] + kh;
        flat = flat * shape[6] + kw;
        return flat;
    }

    static int indexNchw(int n, int c, int h, int w, int channels, int height, int width) {
        return ((n * channels + c) * height + h) * width + w;
    }

    static int indexOihw(int oc, int ic, int kh, int kw, int outC, int inC, int kH, int kW) {
        return ((oc * inC + ic) * kH + kh) * kW + kw;
    }

    static float[] scatterWindowsGrad(float[] gradWindows, int[] paddedShape, int[] im2colArg) {
        int[] winShape = windowShape(paddedShape, new int[] {0, paddedShape[1], im2colArg[0], im2colArg[1]}, 
                Conv2dArg.im2colOnPadded(im2colArg[2], im2colArg[3], im2colArg[8], im2colArg[4], im2colArg[5]),
                new int[] {paddedShape[0], 0, im2colArg[6], im2colArg[7]});
        Conv2dArg cfg = Conv2dArg.im2colOnPadded(im2colArg[2], im2colArg[3], im2colArg[8], im2colArg[4], im2colArg[5]);
        int n = paddedShape[0];
        int inC = paddedShape[1];
        int inH = paddedShape[2];
        int inW = paddedShape[3];
        int groups = im2colArg[8];
        int cinPerGroup = inC / groups;
        int kH = im2colArg[0];
        int kW = im2colArg[1];
        int outH = im2colArg[6];
        int outW = im2colArg[7];
        float[] gradPadded = new float[Strides.numel(paddedShape)];
        for (int batch = 0; batch < n; batch++) {
            for (int group = 0; group < groups; group++) {
                for (int ic = 0; ic < cinPerGroup; ic++) {
                    int channel = group * cinPerGroup + ic;
                    for (int oh = 0; oh < outH; oh++) {
                        for (int ow = 0; ow < outW; ow++) {
                            for (int kh = 0; kh < kH; kh++) {
                                for (int kw = 0; kw < kW; kw++) {
                                    int ih = oh * cfg.strideH + kh * cfg.dilH;
                                    int iw = ow * cfg.strideW + kw * cfg.dilW;
                                    if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                        float g = gradWindows[indexWindow(batch, group, ic, oh, ow, kh, kw, winShape)];
                                        gradPadded[indexNchw(batch, channel, ih, iw, inC, inH, inW)] += g;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return gradPadded;
    }

    public static Tensor scatterWindowsGrad(Tensor gradWindows, int[] paddedShape, int[] im2colArg) {
        return Tensor.of(scatterWindowsGrad(gradWindows.toFlatArray(), paddedShape, im2colArg), paddedShape);
    }
}
