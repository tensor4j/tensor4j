/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.core.lazy;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * GGUF mmap slice for lazy load (tinygrad {@code tensor[data_start+off:]} → {@code ggml_data_to_tensor}).
 * Holds uint8/quant or F32 bytes; {@link #floatShape()} is the output float tensor shape (ggml {@code ne[]}).
 */
public final class LazyGgufSlice {

    public static final int GGML_TYPE_F32 = 0;
    public static final int GGML_TYPE_Q4_0 = 2;

    private final ByteBuffer buffer;
    private final int offset;
    private final int typeId;
    private final int[] floatShape;

    private LazyGgufSlice(ByteBuffer buffer, int offset, int typeId, int[] floatShape) {
        this.buffer = buffer;
        this.offset = offset;
        this.typeId = typeId;
        this.floatShape = floatShape.clone();
    }

    public static LazyGgufSlice q4_0(ByteBuffer buffer, int offset, int[] floatShape) {
        validateBuffer(buffer);
        int numElements = numElements(floatShape, true);
        int need = LazyQuantMath.quantBytes(numElements);
        if (buffer.remaining() - offset < need) {
            throw new IllegalArgumentException("need " + need + " quant bytes at offset " + offset);
        }
        return new LazyGgufSlice(buffer, offset, GGML_TYPE_Q4_0, floatShape);
    }

    public static LazyGgufSlice q4_0(byte[] bytes, int[] floatShape) {
        return q4_0(LazyQuantMath.wrapBytes(bytes), 0, floatShape);
    }

    public static LazyGgufSlice f32(ByteBuffer buffer, int offset, int[] floatShape) {
        validateBuffer(buffer);
        int numElements = numElements(floatShape, false);
        int need = numElements * 4;
        if (buffer.remaining() - offset < need) {
            throw new IllegalArgumentException("need " + need + " f32 bytes at offset " + offset);
        }
        return new LazyGgufSlice(buffer, offset, GGML_TYPE_F32, floatShape);
    }

    public ByteBuffer buffer() {
        return buffer;
    }

    public int offset() {
        return offset;
    }

    public int typeId() {
        return typeId;
    }

    public int[] floatShape() {
        return floatShape.clone();
    }

    public int numElements() {
        return numElements(floatShape, typeId == GGML_TYPE_Q4_0);
    }

    public int storageBytes() {
        return typeId == GGML_TYPE_Q4_0 ? LazyQuantMath.quantBytes(numElements()) : numElements() * 4;
    }

    /** tinygrad Q4 blocks layout: {@code (-1, 18)}. */
    public int[] q4BlockShape() {
        if (typeId != GGML_TYPE_Q4_0) {
            throw new IllegalStateException("not Q4_0");
        }
        int blocks = numElements() / LazyQuantMath.QK4_0;
        return new int[] {blocks, LazyQuantMath.BLOCK_BYTES_Q4_0};
    }

    private static void validateBuffer(ByteBuffer buffer) {
        if (buffer.order() != ByteOrder.LITTLE_ENDIAN) {
            throw new IllegalArgumentException("buffer must be little-endian");
        }
    }

    private static int numElements(int[] shape, boolean q4Aligned) {
        int total = 1;
        for (int dim : shape) {
            if (dim <= 0) {
                throw new IllegalArgumentException("invalid dim " + dim);
            }
            total *= dim;
        }
        if (q4Aligned && total % LazyQuantMath.QK4_0 != 0) {
            throw new IllegalArgumentException("element count " + total + " not divisible by "
                    + LazyQuantMath.QK4_0);
        }
        return total;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof LazyGgufSlice slice && offset == slice.offset && typeId == slice.typeId
                && buffer == slice.buffer && Arrays.equals(floatShape, slice.floatShape);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(new Object[] {buffer, offset, typeId, Arrays.hashCode(floatShape)});
    }
}
