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

import java.util.Arrays;

/** Packed conv2d hyper-parameters (tinygrad {@code conv2d} kwargs subset). */
public final class Conv2dArg {

    /** {@code [strideH, strideW, padBeforeH, padAfterH, padBeforeW, padAfterW, groups, dilH, dilW]}. */
    public static final int ARG_LEN = 9;

    public final int strideH;
    public final int strideW;
    public final int padBeforeH;
    public final int padAfterH;
    public final int padBeforeW;
    public final int padAfterW;
    public final int groups;
    public final int dilH;
    public final int dilW;

    private Conv2dArg(int strideH, int strideW, int padBeforeH, int padAfterH, int padBeforeW, int padAfterW,
            int groups, int dilH, int dilW) {
        this.strideH = strideH;
        this.strideW = strideW;
        this.padBeforeH = padBeforeH;
        this.padAfterH = padAfterH;
        this.padBeforeW = padBeforeW;
        this.padAfterW = padAfterW;
        this.groups = groups;
        this.dilH = dilH;
        this.dilW = dilW;
    }

    public static Conv2dArg defaults() {
        return parse(defaultPacked());
    }

    public static int[] defaultPacked() {
        return new int[] {1, 1, 0, 0, 0, 0, 1, 1, 1};
    }

    public static int[] packed(int stride, int padBeforeH, int padAfterH, int padBeforeW, int padAfterW, int groups) {
        return packed(stride, padBeforeH, padAfterH, padBeforeW, padAfterW, groups, 1);
    }

    public static int[] packed(int stride, int padBeforeH, int padAfterH, int padBeforeW, int padAfterW, int groups,
            int dilation) {
        return new int[] {stride, stride, padBeforeH, padAfterH, padBeforeW, padAfterW, groups, dilation, dilation};
    }

    public static Conv2dArg parse(int[] raw) {
        if (raw.length != 7 && raw.length != ARG_LEN) {
            throw new IllegalArgumentException("conv2d arg length must be 7 or " + ARG_LEN);
        }
        int dilH = raw.length > 7 ? raw[7] : 1;
        int dilW = raw.length > 8 ? raw[8] : 1;
        return new Conv2dArg(raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], dilH, dilW);
    }

    /** Config for {@link ConvIm2Col} on already-padded tensors. */
    public static Conv2dArg im2colOnPadded(int strideH, int strideW, int groups, int dilH, int dilW) {
        return new Conv2dArg(strideH, strideW, 0, 0, 0, 0, groups, dilH, dilW);
    }

    public int[] packed() {
        return new int[] {strideH, strideW, padBeforeH, padAfterH, padBeforeW, padAfterW, groups, dilH, dilW};
    }

    public int effectiveKernel(int kernel) {
        return (kernel - 1) * dilH + 1;
    }

    public int[] padArg(int[] inputNchw) {
        if (inputNchw.length != 4) {
            throw new IllegalArgumentException("conv2d expects NCHW input");
        }
        return new int[] {0, 0, 0, 0, padBeforeH, padAfterH, padBeforeW, padAfterW};
    }

    public int[] outputShape(int[] inputNchw, int[] weightOihw) {
        validateShapes(inputNchw, weightOihw);
        int outC = weightOihw[0];
        int effH = effectiveKernel(weightOihw[2]);
        int effW = effectiveKernel(weightOihw[3]);
        int h = inputNchw[2] + padBeforeH + padAfterH;
        int w = inputNchw[3] + padBeforeW + padAfterW;
        int outH = (h - effH) / strideH + 1;
        int outW = (w - effW) / strideW + 1;
        if (outH < 1 || outW < 1) {
            throw new IllegalArgumentException("conv2d output spatial size invalid");
        }
        return new int[] {inputNchw[0], outC, outH, outW};
    }

    /** IM2COL arg: {@code [kH, kW, strideH, strideW, dilH, dilW, outH, outW, groups]}. */
    public int[] im2colArg(int[] weightOihw, int[] outputNchw) {
        return new int[] {
                weightOihw[2], weightOihw[3],
                strideH, strideW, dilH, dilW,
                outputNchw[2], outputNchw[3], groups
        };
    }

    public boolean canWinograd(int[] weightOihw) {
        return groups == 1 && strideH == 1 && strideW == 1 && dilH == 1 && dilW == 1
                && weightOihw[2] == 3 && weightOihw[3] == 3;
    }

    public void validateShapes(int[] inputNchw, int[] weightOihw) {
        if (inputNchw.length != 4 || weightOihw.length != 4) {
            throw new IllegalArgumentException("conv2d expects NCHW tensors");
        }
        if (groups < 1 || inputNchw[1] % groups != 0 || weightOihw[0] % groups != 0) {
            throw new IllegalArgumentException("invalid conv2d groups " + groups);
        }
        if (inputNchw[1] / groups != weightOihw[1]) {
            throw new IllegalArgumentException("input channels " + inputNchw[1]
                    + " must match weight cin per group " + weightOihw[1]);
        }
    }

    @Override
    public String toString() {
        return "Conv2dArg" + Arrays.toString(packed());
    }
}
