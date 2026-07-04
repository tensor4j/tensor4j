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

import com.github.tensor4j.runtime.infer.InferTensor;

/** Lazy GGUF weight view; dequant reads directly from slice without quant-byte copy. */
public record GgufWeightView(GgufTensorSlice slice, LlamaQkLayout qkLayout, int qkHeads) {

    public GgufWeightView(GgufTensorSlice slice) {
        this(slice, LlamaQkLayout.NONE, 0);
    }

    public float[] dequantizeToFloat() {
        return GgufWeightLoader.dequantizeSlice(slice);
    }

    public InferTensor toMatrix() {
        return toMatrix(qkLayout, qkHeads);
    }

    /**
     * Matrix load with optional llama Q/K permute (tinygrad {@code from_gguf}).
     */
    public InferTensor toMatrix(LlamaQkLayout layout, int nHeads) {
        float[] data = dequantizeToFloat();
        long[] ne = slice.shape().ne();
        int rows = (int) ne[0];
        int cols = slice.shape().rank() >= 2 ? (int) ne[1] : 1;
        if (layout == LlamaQkLayout.PERMUTE_QK && nHeads > 0) {
            GgufLlamaLayout.permuteQkInterleaved(data, rows, cols, nHeads);
        } else if (layout == LlamaQkLayout.PERMUTE_QK_KV && nHeads > 0) {
            rows = cols;
            cols = (int) ne[0];
            GgufLlamaLayout.permuteQkInterleaved(data, rows, cols, nHeads);
        } else if (layout == LlamaQkLayout.REVERSE_GGUF_DIMS) {
            rows = cols;
            cols = (int) ne[0];
        }
        return InferTensor.of(data, rows, cols);
    }

    public InferTensor toEmbedding() {
        return GgufWeightLoader.fromGgmlEmbedding(slice.shape(), dequantizeToFloat());
    }

    public InferTensor toVector() {
        return GgufWeightLoader.fromGgmlVector(slice.shape(), dequantizeToFloat());
    }
}
