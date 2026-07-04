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

/**
 * Q4_0 dequant math for lazy graphs (tinygrad {@code ggml_data_to_tensor} type 2).
 * Kept in core/lazy so {@link LazyUOp} realize does not depend on runtime/gguf.
 */
final class LazyQuantMath {

    static final int QK4_0 = 32;
    static final int BLOCK_BYTES_Q4_0 = 18;

    private LazyQuantMath() {
    }

    static float[] dequantizeQ4_0(ByteBuffer buffer, int offset, int numElements) {
        if (numElements % QK4_0 != 0) {
            throw new IllegalArgumentException("numElements " + numElements + " not divisible by " + QK4_0);
        }
        int numBlocks = numElements / QK4_0;
        float[] out = new float[numElements];
        int pos = offset;
        for (int block = 0; block < numBlocks; block++) {
            float scale = readFp16(buffer, pos);
            pos += 2;
            for (int j = 0; j < QK4_0 / 2; j++) {
                int packed = buffer.get(pos++) & 0xFF;
                int q0 = (packed & 0x0F) - 8;
                int q1 = (packed >>> 4) - 8;
                int base = block * QK4_0 + j;
                out[base] = q0 * scale;
                out[base + QK4_0 / 2] = q1 * scale;
            }
        }
        return out;
    }

    static int quantBytes(int numElements) {
        if (numElements % QK4_0 != 0) {
            throw new IllegalArgumentException("numElements " + numElements + " not divisible by " + QK4_0);
        }
        return numElements / QK4_0 * BLOCK_BYTES_Q4_0;
    }

    private static float readFp16(ByteBuffer buffer, int offset) {
        int bits = (buffer.get(offset) & 0xFF) | ((buffer.get(offset + 1) & 0xFF) << 8);
        return fp16ToFloat32(bits);
    }

    /** IEEE fp16 → float32 (ggml block scale). */
    private static float fp16ToFloat32(int bits) {
        int sign = (bits >>> 15) & 0x1;
        int exp = (bits >>> 10) & 0x1F;
        int mant = bits & 0x3FF;
        if (exp == 0) {
            if (mant == 0) {
                return sign == 0 ? 0.0f : -0.0f;
            }
            return (float) ((sign == 0 ? 1 : -1) * Math.pow(2, -14) * (mant / 1024.0));
        }
        if (exp == 31) {
            return mant == 0 ? (sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY) : Float.NaN;
        }
        return (float) ((sign == 0 ? 1 : -1) * Math.pow(2, exp - 15) * (1.0 + mant / 1024.0));
    }

    static float[] readF32(ByteBuffer buffer, int offset, int count) {
        float[] out = new float[count];
        for (int i = 0; i < count; i++) {
            out[i] = buffer.getFloat(offset + i * 4);
        }
        return out;
    }

    static ByteBuffer wrapBytes(byte[] bytes) {
        return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    }
}
