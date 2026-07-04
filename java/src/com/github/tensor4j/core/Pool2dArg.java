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

/** Packed pool2d hyper-parameters (tinygrad {@code max_pool2d} / {@code avg_pool2d} subset). */
public final class Pool2dArg {

    /** {@code [kH, kW, strideH, strideW, padBeforeH, padAfterH, padBeforeW, padAfterW, mode]}. */
    public static final int ARG_LEN = 9;

    public static final int MODE_MAX = 0;
    public static final int MODE_AVG = 1;

    public final int kernelH;
    public final int kernelW;
    public final int strideH;
    public final int strideW;
    public final int padBeforeH;
    public final int padAfterH;
    public final int padBeforeW;
    public final int padAfterW;
    public final int mode;

    private Pool2dArg(int kernelH, int kernelW, int strideH, int strideW, int padBeforeH, int padAfterH,
            int padBeforeW, int padAfterW, int mode) {
        this.kernelH = kernelH;
        this.kernelW = kernelW;
        this.strideH = strideH;
        this.strideW = strideW;
        this.padBeforeH = padBeforeH;
        this.padAfterH = padAfterH;
        this.padBeforeW = padBeforeW;
        this.padAfterW = padAfterW;
        this.mode = mode;
    }

    public static Pool2dArg parse(int[] raw) {
        if (raw.length != ARG_LEN) {
            throw new IllegalArgumentException("pool2d arg length must be " + ARG_LEN);
        }
        return new Pool2dArg(raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], raw[8]);
    }

    public static int[] maxPacked(int kernel, int stride) {
        return packed(kernel, kernel, stride, stride, 0, 0, 0, 0, MODE_MAX);
    }

    public static int[] avgPacked(int kernel, int stride) {
        return packed(kernel, kernel, stride, stride, 0, 0, 0, 0, MODE_AVG);
    }

    public static int[] packed(int kH, int kW, int strideH, int strideW, int padBeforeH, int padAfterH,
            int padBeforeW, int padAfterW, int mode) {
        return new int[] {kH, kW, strideH, strideW, padBeforeH, padAfterH, padBeforeW, padAfterW, mode};
    }

    public int[] packed() {
        return new int[] {
                kernelH, kernelW, strideH, strideW, padBeforeH, padAfterH, padBeforeW, padAfterW, mode
        };
    }

    public int[] padArg(int[] inputNchw) {
        if (inputNchw.length != 4) {
            throw new IllegalArgumentException("pool2d expects NCHW input");
        }
        return new int[] {0, 0, 0, 0, padBeforeH, padAfterH, padBeforeW, padAfterW};
    }

    public int[] outputShape(int[] inputNchw) {
        if (inputNchw.length != 4) {
            throw new IllegalArgumentException("pool2d expects NCHW input");
        }
        int h = inputNchw[2] + padBeforeH + padAfterH;
        int w = inputNchw[3] + padBeforeW + padAfterW;
        int outH = (h - kernelH) / strideH + 1;
        int outW = (w - kernelW) / strideW + 1;
        if (outH < 1 || outW < 1) {
            throw new IllegalArgumentException("pool2d output spatial size invalid");
        }
        return new int[] {inputNchw[0], inputNchw[1], outH, outW};
    }

    /** IM2COL-style window arg: {@code [kH, kW, strideH, strideW, 1, 1, outH, outW, 1]}. */
    public int[] windowArg(int[] outputNchw) {
        return new int[] {
                kernelH, kernelW, strideH, strideW, 1, 1, outputNchw[2], outputNchw[3], 1
        };
    }

    public boolean isMax() {
        return mode == MODE_MAX;
    }

    @Override
    public String toString() {
        return "Pool2dArg" + Arrays.toString(packed());
    }
}
