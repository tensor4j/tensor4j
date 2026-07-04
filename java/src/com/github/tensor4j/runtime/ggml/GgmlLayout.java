/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.ggml;

/** ggml byte layout ({@code nb[]} strides and {@code ggml_nbytes}). */
public final class GgmlLayout {

    private GgmlLayout() {
    }

    /** {@code ggml_nbytes}: type_size * nelements / block_size. */
    public static long numBytes(GgmlType type, GgmlTensorShape shape) {
        validateRowDivisible(type, shape.ne(0));
        long blocks = shape.numElements() / type.blockSize();
        return type.typeSizeBytes() * blocks;
    }

    /** {@code nb[0..3]} byte strides (gguf tensor info layout). */
    public static long[] byteStrides(GgmlType type, GgmlTensorShape shape) {
        long[] ne = shape.ne();
        validateRowDivisible(type, ne[0]);
        long[] nb = new long[GgmlTensorShape.MAX_DIMS];
        nb[0] = type.typeSizeBytes();
        nb[1] = nb[0] * (ne[0] / type.blockSize());
        for (int j = 2; j < GgmlTensorShape.MAX_DIMS; j++) {
            nb[j] = nb[j - 1] * ne[j - 1];
        }
        return nb;
    }

    /** Pad offset to alignment ({@code GGML_PAD}). */
    public static long pad(long offset, int alignment) {
        if (alignment <= 0) {
            throw new IllegalArgumentException("alignment must be positive");
        }
        long remainder = offset % alignment;
        if (remainder == 0) {
            return offset;
        }
        return offset + alignment - remainder;
    }

    private static void validateRowDivisible(GgmlType type, long rowElements) {
        if (type.blockSize() == 0 || rowElements % type.blockSize() != 0) {
            throw new IllegalArgumentException("row elements " + rowElements
                    + " not divisible by block size " + type.blockSize() + " for " + type);
        }
    }
}
