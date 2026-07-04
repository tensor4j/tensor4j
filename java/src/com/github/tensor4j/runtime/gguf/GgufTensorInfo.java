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

import com.github.tensor4j.runtime.ggml.GgmlLayout;
import com.github.tensor4j.runtime.ggml.GgmlTensorShape;
import com.github.tensor4j.runtime.ggml.GgmlType;

/** Tensor descriptor from GGUF metadata section. */
public record GgufTensorInfo(String name, GgmlTensorShape shape, GgmlType type, long offsetBytes, long sizeBytes,
        long[] byteStrides) {

    public static GgufTensorInfo parse(String name, int nDims, long[] dims, GgmlType type, long offsetBytes) {
        GgmlTensorShape shape = GgmlTensorShape.fromGguf(nDims, dims);
        long sizeBytes = GgmlLayout.numBytes(type, shape);
        long[] byteStrides = GgmlLayout.byteStrides(type, shape);
        return new GgufTensorInfo(name, shape, type, offsetBytes, sizeBytes, byteStrides);
    }
}
