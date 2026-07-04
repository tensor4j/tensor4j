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

/** max_unpool2d (tinygrad {@code max_unpool2d} teaching subset). */
public final class MaxUnpool2dMath {

    private MaxUnpool2dMath() {
    }

    public static Tensor forward(Tensor values, Tensor indices, int[] poolArg, int[] outputShape) {
        Pool2dArg cfg = Pool2dArg.parse(poolArg);
        int[] valShape = values.shape().dims();
        int n = outputShape[0];
        int c = outputShape[1];
        int inH = outputShape[2];
        int inW = outputShape[3];
        int outH = valShape[2];
        int outW = valShape[3];
        float[] out = new float[Strides.numel(outputShape)];
        float[] vals = values.toFlatArray();
        float[] idxFlat = indices.toFlatArray();
        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < c; channel++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        int poolIndex = ConvIm2Col.indexNchw(batch, channel, oh, ow, c, outH, outW);
                        int spatialFlat = (int) idxFlat[poolIndex];
                        int ih = spatialFlat / inW;
                        int iw = spatialFlat % inW;
                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                            out[ConvIm2Col.indexNchw(batch, channel, ih, iw, c, inH, inW)] = vals[poolIndex];
                        }
                    }
                }
            }
        }
        return Tensor.of(out, outputShape);
    }

    public static Tensor gradValues(Tensor gradOutput, Tensor indices, int[] poolArg, int[] valueShape) {
        Pool2dArg cfg = Pool2dArg.parse(poolArg);
        int[] outShape = valueShape;
        int n = outShape[0];
        int c = outShape[1];
        int poolH = outShape[2];
        int poolW = outShape[3];
        int inH = gradOutput.shape().dims()[2];
        int inW = gradOutput.shape().dims()[3];
        float[] gradVals = new float[Strides.numel(outShape)];
        float[] gradOut = gradOutput.toFlatArray();
        float[] idxFlat = indices.toFlatArray();
        for (int batch = 0; batch < n; batch++) {
            for (int channel = 0; channel < c; channel++) {
                for (int oh = 0; oh < poolH; oh++) {
                    for (int ow = 0; ow < poolW; ow++) {
                        int poolIndex = ConvIm2Col.indexNchw(batch, channel, oh, ow, c, poolH, poolW);
                        int spatialFlat = (int) idxFlat[poolIndex];
                        int ih = spatialFlat / inW;
                        int iw = spatialFlat % inW;
                        if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                            gradVals[poolIndex] = gradOut[ConvIm2Col.indexNchw(batch, channel, ih, iw, c, inH, inW)];
                        }
                    }
                }
            }
        }
        return Tensor.of(gradVals, outShape);
    }
}
