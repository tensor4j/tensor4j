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
 * Winograd F(4,3) conv2d for 3x3 stride-1 (tinygrad {@code conv2d} Winograd branch).
 */
public final class ConvWinograd {

    private static final int TILE = 4;
    private static final int TRANS = 6;

    private static final float[][] G = {
            {1f / 4f, 0f, 0f},
            {-1f / 6f, -1f / 6f, -1f / 6f},
            {-1f / 6f, 1f / 6f, -1f / 6f},
            {1f / 24f, 1f / 12f, 1f / 6f},
            {1f / 24f, -1f / 12f, 1f / 6f},
            {0f, 0f, 1f}
    };

    private static final float[][] BT = {
            {4f, 0f, -5f, 0f, 1f, 0f},
            {0f, -4f, -4f, 1f, 1f, 0f},
            {0f, 4f, -4f, -1f, 1f, 0f},
            {0f, -2f, -1f, 2f, 1f, 0f},
            {0f, 2f, -1f, -2f, 1f, 0f},
            {0f, 4f, 0f, -5f, 0f, 1f}
    };

    private static final float[][] AT = {
            {1f, 1f, 1f, 1f, 1f, 0f},
            {0f, 1f, -1f, 2f, -2f, 0f},
            {0f, 1f, 1f, 4f, 4f, 0f},
            {0f, 1f, -1f, 8f, -8f, 1f}
    };

    private ConvWinograd() {
    }

    public static Tensor forward(Tensor input, Tensor weight, Conv2dArg cfg) {
        int[] inShape = input.shape().dims();
        int[] wShape = weight.shape().dims();
        if (!cfg.canWinograd(wShape)) {
            throw new IllegalArgumentException("Winograd requires 3x3 stride-1 dilation-1 groups-1");
        }
        cfg.validateShapes(inShape, wShape);
        Tensor padded = PadShrink.applyPad(input, cfg.padArg(inShape));
        int[] outShape = cfg.outputShape(inShape, wShape);
        float[] out = new float[Strides.numel(outShape)];
        float[] inFlat = padded.toFlatArray();
        int[] padShape = padded.shape().dims();
        int n = inShape[0];
        int inC = inShape[1];
        int outC = wShape[0];
        int outH = outShape[2];
        int outW = outShape[3];
        float[][][][] uFilter = transformFilters(weight.toFlatArray(), wShape);
        for (int batch = 0; batch < n; batch++) {
            for (int oc = 0; oc < outC; oc++) {
                for (int tileY = 0; tileY < outH; tileY += TILE) {
                    for (int tileX = 0; tileX < outW; tileX += TILE) {
                        float[][] tileOut = new float[TILE][TILE];
                        for (int ic = 0; ic < inC; ic++) {
                            float[][] v = transformInputTile(inFlat, padShape, batch, ic, tileY, tileX);
                            float[][] u = uFilter[oc][ic];
                            float[][] hadamard = hadamard6(u, v);
                            float[][] contrib = matMul(matMul(AT, hadamard), transpose(AT));
                            addTile(tileOut, contrib);
                        }
                        writeOutputTile(out, outShape, batch, oc, tileY, tileX, tileOut);
                    }
                }
            }
        }
        return Tensor.of(out, outShape);
    }

    private static float[][][][] transformFilters(float[] weight, int[] wShape) {
        int outC = wShape[0];
        int inC = wShape[1];
        float[][][][] transformed = new float[outC][inC][TRANS][TRANS];
        for (int oc = 0; oc < outC; oc++) {
            for (int ic = 0; ic < inC; ic++) {
                float[][] kernel = new float[3][3];
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        kernel[kh][kw] = weight[ConvIm2Col.indexOihw(oc, ic, kh, kw, outC, inC, 3, 3)];
                    }
                }
                transformed[oc][ic] = matMul(G, matMul(kernel, transpose(G)));
            }
        }
        return transformed;
    }

    private static float[][] transformInputTile(float[] input, int[] shape, int batch, int channel, int tileY,
            int tileX) {
        float[][] patch = new float[TRANS][TRANS];
        for (int y = 0; y < TRANS; y++) {
            for (int x = 0; x < TRANS; x++) {
                int ih = tileY + y;
                int iw = tileX + x;
                if (ih >= 0 && ih < shape[2] && iw >= 0 && iw < shape[3]) {
                    patch[y][x] = input[ConvIm2Col.indexNchw(batch, channel, ih, iw, shape[1], shape[2], shape[3])];
                }
            }
        }
        return matMul(BT, matMul(patch, transpose(BT)));
    }

    private static void addTile(float[][] target, float[][] contrib) {
        for (int y = 0; y < TILE; y++) {
            for (int x = 0; x < TILE; x++) {
                target[y][x] += contrib[y][x];
            }
        }
    }

    private static void writeOutputTile(float[] out, int[] outShape, int batch, int oc, int tileY, int tileX,
            float[][] tile) {
        int outH = outShape[2];
        int outW = outShape[3];
        for (int y = 0; y < TILE; y++) {
            for (int x = 0; x < TILE; x++) {
                int oh = tileY + y;
                int ow = tileX + x;
                if (oh < outH && ow < outW) {
                    out[ConvIm2Col.indexNchw(batch, oc, oh, ow, outShape[1], outH, outW)] = tile[y][x];
                }
            }
        }
    }

    private static float[][] hadamard6(float[][] left, float[][] right) {
        float[][] out = new float[TRANS][TRANS];
        for (int i = 0; i < TRANS; i++) {
            for (int j = 0; j < TRANS; j++) {
                out[i][j] = left[i][j] * right[i][j];
            }
        }
        return out;
    }

    private static float[][] matMul(float[][] left, float[][] right) {
        int rows = left.length;
        int cols = right[0].length;
        int inner = right.length;
        float[][] out = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float sum = 0f;
                for (int k = 0; k < inner; k++) {
                    sum += left[i][k] * right[k][j];
                }
                out[i][j] = sum;
            }
        }
        return out;
    }

    private static float[][] transpose(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[][] out = new float[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[j][i] = matrix[i][j];
            }
        }
        return out;
    }
}
