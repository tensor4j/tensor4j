/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.gguf;

/**
 * Llama-arch GGUF weight layout fixes (tinygrad {@code Transformer.from_gguf}).
 *
 * <p>Q/K matrices use interleaved RoPE row layout in GGUF; tinygrad applies
 * {@code rearrange("(n h two) d -> (n two h) d")} on load.
 */
public final class GgufLlamaLayout {

    private GgufLlamaLayout() {
    }

    /**
     * Permute Q/K rows from GGUF interleaved layout to matmul layout.
     * Public for testability — parity vs tinygrad {@code from_gguf} rearrange.
     *
     * @param data row-major {@code [rows, cols]} ggml flat buffer (mutated in place)
     * @param rows output dim ({@code n_embd} or {@code n_head_kv * head_dim})
     * @param nHeads head count for this matrix (Q: {@code n_head}, K: {@code n_head_kv})
     */
    public static void permuteQkInterleaved(float[] data, int rows, int cols, int nHeads) {
        if (rows % nHeads != 0) {
            throw new IllegalArgumentException("rows " + rows + " not divisible by nHeads " + nHeads);
        }
        int headDim = rows / nHeads;
        if (headDim % 2 != 0) {
            throw new IllegalArgumentException("headDim " + headDim + " must be even");
        }
        int h = headDim / 2;
        float[] scratch = data.clone();
        for (int col = 0; col < cols; col++) {
            for (int n = 0; n < nHeads; n++) {
                for (int two = 0; two < 2; two++) {
                    for (int hi = 0; hi < h; hi++) {
                        int srcRow = n * headDim + hi * 2 + two;
                        int dstRow = n * headDim + two * h + hi;
                        data[dstRow * cols + col] = scratch[srcRow * cols + col];
                    }
                }
            }
        }
    }

    /**
     * Reinterpret ggml row-major {@code [d0, d1]} as tinygrad {@code reshape(*reversed(dims))} → {@code [d1, d0]}.
     */
    public static float[] reverseGgufDims(float[] data, int d0, int d1) {
        if (data.length != d0 * d1) {
            throw new IllegalArgumentException("data length " + data.length + " != " + d0 + "*" + d1);
        }
        float[] out = new float[data.length];
        for (int r1 = 0; r1 < d1; r1++) {
            for (int c1 = 0; c1 < d0; c1++) {
                int p = r1 * d0 + c1;
                int r0 = p / d1;
                int c0 = p % d1;
                out[p] = data[r0 * d1 + c0];
            }
        }
        return out;
    }
}
