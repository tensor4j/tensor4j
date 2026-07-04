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

/** Adaptive pool2d arg: {@code [outH, outW, mode]}. */
public final class AdaptivePool2dArg {

    public static final int ARG_LEN = 3;

    public final int outH;
    public final int outW;
    public final int mode;

    private AdaptivePool2dArg(int outH, int outW, int mode) {
        this.outH = outH;
        this.outW = outW;
        this.mode = mode;
    }

    public static AdaptivePool2dArg parse(int[] raw) {
        if (raw.length != ARG_LEN) {
            throw new IllegalArgumentException("adaptive pool arg length must be " + ARG_LEN);
        }
        return new AdaptivePool2dArg(raw[0], raw[1], raw[2]);
    }

    public static int[] packed(int outH, int outW, int mode) {
        return new int[] {outH, outW, mode};
    }

    public int[] outputShape(int[] inputNchw) {
        return new int[] {inputNchw[0], inputNchw[1], outH, outW};
    }

    /** Equivalent fixed-kernel pool arg (PyTorch adaptive formula). */
    public int[] poolArg(int[] inputNchw) {
        int inH = inputNchw[2];
        int inW = inputNchw[3];
        int strideH = inH / outH;
        int strideW = inW / outW;
        int kernelH = inH - (outH - 1) * strideH;
        int kernelW = inW - (outW - 1) * strideW;
        return Pool2dArg.packed(kernelH, kernelW, strideH, strideW, 0, 0, 0, 0, mode);
    }
}
