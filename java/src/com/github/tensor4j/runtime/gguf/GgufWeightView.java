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
public record GgufWeightView(GgufTensorSlice slice) {

    public float[] dequantizeToFloat() {
        return GgufWeightLoader.dequantizeSlice(slice);
    }

    public InferTensor toMatrix() {
        return GgufWeightLoader.fromGgmlMatrix(slice.shape(), dequantizeToFloat());
    }

    public InferTensor toEmbedding() {
        return GgufWeightLoader.fromGgmlEmbedding(slice.shape(), dequantizeToFloat());
    }

    public InferTensor toVector() {
        return GgufWeightLoader.fromGgmlVector(slice.shape(), dequantizeToFloat());
    }
}
