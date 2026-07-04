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

/** Packed conv_transpose2d hyper-parameters (tinygrad subset). */
public final class ConvTranspose2dArg {

    /** {@code [strideH, strideW, padBeforeH, padAfterH, padBeforeW, padAfterW, groups, outPadH, outPadW]}. */
    public static final int ARG_LEN = 9;

    public final int strideH;
    public final int strideW;
    public final int padBeforeH;
    public final int padAfterH;
    public final int padBeforeW;
    public final int padAfterW;
    public final int groups;
    public final int outPadH;
    public final int outPadW;

    private ConvTranspose2dArg(int strideH, int strideW, int padBeforeH, int padAfterH, int padBeforeW, int padAfterW,
            int groups, int outPadH, int outPadW) {
        this.strideH = strideH;
        this.strideW = strideW;
        this.padBeforeH = padBeforeH;
        this.padAfterH = padAfterH;
        this.padBeforeW = padBeforeW;
        this.padAfterW = padAfterW;
        this.groups = groups;
        this.outPadH = outPadH;
        this.outPadW = outPadW;
    }

    public static ConvTranspose2dArg defaults() {
        return parse(defaultPacked());
    }

    public static int[] defaultPacked() {
        return new int[] {1, 1, 0, 0, 0, 0, 1, 0, 0};
    }

    public static int[] packed(int stride, int padBeforeH, int padAfterH, int padBeforeW, int padAfterW, int groups) {
        return new int[] {stride, stride, padBeforeH, padAfterH, padBeforeW, padAfterW, groups, 0, 0};
    }

    public static ConvTranspose2dArg parse(int[] raw) {
        if (raw.length != ARG_LEN) {
            throw new IllegalArgumentException("conv_transpose2d arg length must be " + ARG_LEN);
        }
        return new ConvTranspose2dArg(raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], raw[8]);
    }

    public int[] packed() {
        return new int[] {
                strideH, strideW, padBeforeH, padAfterH, padBeforeW, padAfterW, groups, outPadH, outPadW
        };
    }

    public void validateShapes(int[] inputNchw, int[] weightIohw) {
        if (inputNchw.length != 4 || weightIohw.length != 4) {
            throw new IllegalArgumentException("conv_transpose2d expects NCHW tensors");
        }
        if (groups < 1 || inputNchw[1] % groups != 0) {
            throw new IllegalArgumentException("invalid conv_transpose2d groups " + groups);
        }
        int coutPerGroup = weightIohw[1];
        if (coutPerGroup < 1) {
            throw new IllegalArgumentException("weight cout per group invalid");
        }
        if (weightIohw[0] != inputNchw[1]) {
            throw new IllegalArgumentException("weight cin " + weightIohw[0]
                    + " must match input channels " + inputNchw[1]);
        }
    }

    public int[] outputShape(int[] inputNchw, int[] weightIohw) {
        validateShapes(inputNchw, weightIohw);
        int kH = weightIohw[2];
        int kW = weightIohw[3];
        int outH = (inputNchw[2] - 1) * strideH - padBeforeH - padAfterH + kH + outPadH;
        int outW = (inputNchw[3] - 1) * strideW - padBeforeW - padAfterW + kW + outPadW;
        if (outH < 1 || outW < 1) {
            throw new IllegalArgumentException("conv_transpose2d output spatial size invalid");
        }
        int outC = weightIohw[1] * groups;
        return new int[] {inputNchw[0], outC, outH, outW};
    }
}
