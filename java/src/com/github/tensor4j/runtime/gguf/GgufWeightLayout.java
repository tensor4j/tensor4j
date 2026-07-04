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
import java.util.ArrayList;
import java.util.List;

/** GGUF data-section layout (gguf.cpp tensor offset assignment). */
public final class GgufWeightLayout {

    private GgufWeightLayout() {
    }

    /** Assign sequential offsets: first tensor at 0, each next padded to alignment. */
    public static List<GgufTensorInfo> plan(List<GgufTensorPayload> payloads, int alignment) {
        if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
            throw new IllegalArgumentException("alignment must be a power of two: " + alignment);
        }
        List<GgufTensorInfo> tensors = new ArrayList<>(payloads.size());
        long offset = 0;
        for (GgufTensorPayload payload : payloads) {
            GgmlTensorShape shape = payload.shape();
            GgmlType type = payload.type();
            long sizeBytes = GgmlLayout.numBytes(type, shape);
            long[] byteStrides = GgmlLayout.byteStrides(type, shape);
            tensors.add(new GgufTensorInfo(payload.name(), shape, type, offset, sizeBytes, byteStrides));
            offset += GgmlLayout.pad(sizeBytes, alignment);
        }
        return tensors;
    }

    /** Total bytes in the data section including inter-tensor padding. */
    public static long totalDataBytes(List<GgufTensorInfo> tensors, int alignment) {
        if (tensors.isEmpty()) {
            return 0;
        }
        GgufTensorInfo last = tensors.get(tensors.size() - 1);
        return last.offsetBytes() + GgmlLayout.pad(last.sizeBytes(), alignment);
    }
}
