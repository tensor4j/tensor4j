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

/** Quantized block helpers (ggml-quants.c). */
public final class GgmlQuant {

    public static final int QK4_0 = 32;
    public static final int BLOCK_BYTES_Q4_0 = 18;

    private GgmlQuant() {
    }

    /** {@code dequantize_row_q4_0} — row-major flat layout. */
    public static void dequantizeRowQ4_0(byte[] blocks, int blockOffset, float[] out, int outOffset,
            int numElements) {
        if (numElements % QK4_0 != 0) {
            throw new IllegalArgumentException("numElements " + numElements + " not divisible by " + QK4_0);
        }
        int numBlocks = numElements / QK4_0;
        int pos = blockOffset;
        for (int block = 0; block < numBlocks; block++) {
            float scale = GgmlFp16.toFloat32(blocks, pos);
            pos += 2;
            for (int j = 0; j < QK4_0 / 2; j++) {
                int packed = blocks[pos++] & 0xFF;
                int q0 = (packed & 0x0F) - 8;
                int q1 = (packed >>> 4) - 8;
                int base = outOffset + block * QK4_0 + j;
                out[base] = q0 * scale;
                out[base + QK4_0 / 2] = q1 * scale;
            }
        }
    }

    public static float[] dequantizeQ4_0(byte[] weightBytes, GgmlTensorShape shape) {
        int numElements = (int) shape.numElements();
        float[] out = new float[numElements];
        dequantizeRowQ4_0(weightBytes, 0, out, 0, numElements);
        return out;
    }

    /** Dequant Q4_0 directly from a mmap buffer slice (no quant-byte[] copy). */
    public static float[] dequantizeQ4_0(java.nio.ByteBuffer buffer, int offset, GgmlTensorShape shape) {
        int numElements = (int) shape.numElements();
        float[] out = new float[numElements];
        int numBlocks = numElements / QK4_0;
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

    private static float readFp16(java.nio.ByteBuffer buffer, int offset) {
        int bits = (buffer.get(offset) & 0xFF) | ((buffer.get(offset + 1) & 0xFF) << 8);
        return GgmlFp16.toFloat32(bits);
    }

    /** {@code quantize_row_q4_0_ref} — builds weight bytes for tests and fixtures. */
    public static byte[] quantizeRowQ4_0Reference(float[] values) {
        if (values.length % QK4_0 != 0) {
            throw new IllegalArgumentException("length " + values.length + " not divisible by " + QK4_0);
        }
        int numBlocks = values.length / QK4_0;
        byte[] blocks = new byte[numBlocks * BLOCK_BYTES_Q4_0];
        int pos = 0;
        for (int block = 0; block < numBlocks; block++) {
            float amax = 0.0f;
            float max = 0.0f;
            int blockStart = block * QK4_0;
            for (int j = 0; j < QK4_0; j++) {
                float v = values[blockStart + j];
                float abs = Math.abs(v);
                if (amax < abs) {
                    amax = abs;
                    max = v;
                }
            }
            float scale = max / -8.0f;
            float invScale = scale != 0.0f ? 1.0f / scale : 0.0f;
            GgmlFp16.writeLittleEndian(blocks, pos, GgmlFp16.fromFloat32(scale));
            pos += 2;
            for (int j = 0; j < QK4_0 / 2; j++) {
                float x0 = values[blockStart + j] * invScale;
                float x1 = values[blockStart + QK4_0 / 2 + j] * invScale;
                int xi0 = Math.min(15, (int) (x0 + 8.5f));
                int xi1 = Math.min(15, (int) (x1 + 8.5f));
                blocks[pos++] = (byte) (xi0 | (xi1 << 4));
            }
        }
        return blocks;
    }
}
