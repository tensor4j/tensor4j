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

import com.github.tensor4j.runtime.ggml.GgmlQuant;
import com.github.tensor4j.runtime.ggml.GgmlTensorShape;
import com.github.tensor4j.runtime.ggml.GgmlType;
import com.github.tensor4j.runtime.infer.InferTensor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Load GGUF tensors into {@link InferTensor} (ggml weight layout). */
public final class GgufWeightLoader {

    private GgufWeightLoader() {
    }

    public static GgufWeightView loadView(GgufTensorSource source, String name) {
        return loadView(source, name, LlamaQkLayout.NONE, 0);
    }

    public static GgufWeightView loadView(
            GgufTensorSource source, String name, LlamaQkLayout qkLayout, int nHeads) {
        return new GgufWeightView(source.tensorSlice(name), qkLayout, nHeads);
    }

    public static InferTensor loadMatrix(GgufFile file, String name) {
        return loadMatrix((GgufTensorSource) file, name);
    }

    public static InferTensor loadMatrix(GgufTensorSource source, String name) {
        return loadView(source, name).toMatrix();
    }

    public static InferTensor loadEmbedding(GgufFile file, String name) {
        return loadEmbedding((GgufTensorSource) file, name);
    }

    public static InferTensor loadEmbedding(GgufTensorSource source, String name) {
        return loadView(source, name).toEmbedding();
    }

    public static InferTensor loadVector(GgufFile file, String name) {
        return loadVector((GgufTensorSource) file, name);
    }

    public static InferTensor loadVector(GgufTensorSource source, String name) {
        return loadView(source, name).toVector();
    }

    public static float[] dequantize(GgufFile file, GgufTensorInfo info) {
        return dequantize((GgufTensorSource) file, info);
    }

    public static float[] dequantize(GgufTensorSource source, GgufTensorInfo info) {
        return dequantizeSlice(source.tensorSlice(info.name()));
    }

    public static float[] dequantizeSlice(GgufTensorSlice slice) {
        if (slice.type() == GgmlType.F32) {
            return readF32Slice(slice);
        }
        if (slice.type() == GgmlType.Q4_0) {
            return GgmlQuant.dequantizeQ4_0(slice.buffer(), slice.offset(), slice.shape());
        }
        if (slice.type() == GgmlType.F16) {
            return GgmlQuant.dequantizeF16(slice.buffer(), slice.offset(), slice.shape());
        }
        if (slice.type() == GgmlType.Q4_K) {
            return GgmlQuant.dequantizeQ4_K(slice.buffer(), slice.offset(), slice.shape());
        }
        if (slice.type() == GgmlType.Q6_K) {
            return GgmlQuant.dequantizeQ6_K(slice.buffer(), slice.offset(), slice.shape());
        }
        throw new IllegalArgumentException("unsupported tensor type " + slice.type() + " for " + slice.name());
    }

    static InferTensor fromGgmlMatrix(GgmlTensorShape shape, float[] ggmlFlat) {
        long[] ne = shape.ne();
        int rank = shape.rank();
        if (rank == 1) {
            return InferTensor.vector(ggmlFlat);
        }
        if (rank == 2) {
            int rows = (int) ne[0];
            int cols = (int) ne[1];
            return InferTensor.of(ggmlFlat, rows, cols);
        }
        throw new IllegalArgumentException("unsupported matrix rank " + rank);
    }

    static InferTensor fromGgmlEmbedding(GgmlTensorShape shape, float[] ggmlFlat) {
        long[] ne = shape.ne();
        int nEmbd = (int) ne[0];
        int nVocab = (int) ne[1];
        float[] table = new float[nVocab * nEmbd];
        for (int vocab = 0; vocab < nVocab; vocab++) {
            for (int dim = 0; dim < nEmbd; dim++) {
                table[vocab * nEmbd + dim] = ggmlFlat[dim + vocab * nEmbd];
            }
        }
        return InferTensor.of(table, nVocab, nEmbd);
    }

    static InferTensor fromGgmlVector(GgmlTensorShape shape, float[] ggmlFlat) {
        if (shape.rank() != 1) {
            return fromGgmlMatrix(shape, ggmlFlat);
        }
        return InferTensor.vector(ggmlFlat);
    }

    private static float[] readF32Slice(GgufTensorSlice slice) {
        ByteBuffer buffer = slice.buffer();
        int offset = slice.offset();
        int count = slice.sizeBytes() / 4;
        float[] out = new float[count];
        for (int i = 0; i < count; i++) {
            out[i] = buffer.getFloat(offset + i * 4);
        }
        return out;
    }
}
