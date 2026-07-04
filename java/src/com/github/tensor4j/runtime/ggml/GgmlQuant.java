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
    public static final int QK_K = 256;
    public static final int BLOCK_BYTES_Q4_K = 144;
    public static final int BLOCK_BYTES_Q6_K = 210;

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

    /** {@code dequantize_row_q4_K} — ggml K-quant super-block (256 elems / 144 bytes). */
    public static float[] dequantizeQ4_K(java.nio.ByteBuffer buffer, int offset, GgmlTensorShape shape) {
        int numElements = (int) shape.numElements();
        if (numElements % QK_K != 0) {
            throw new IllegalArgumentException("Q4_K elements " + numElements + " not divisible by " + QK_K);
        }
        float[] out = new float[numElements];
        int numBlocks = numElements / QK_K;
        int pos = offset;
        for (int block = 0; block < numBlocks; block++) {
            float d = readFp16(buffer, pos);
            float dmin = readFp16(buffer, pos + 2);
            float[] sc = new float[8];
            float[] mn = new float[8];
            unpackQ4KScales(buffer, pos + 4, sc, mn);
            int qsPos = pos + 16;
            for (int group = 0; group < 8; group++) {
                float scale = d * sc[group];
                float bias = dmin * mn[group];
                int chunk = group / 2;
                int nibble = group % 2;
                for (int j = 0; j < 32; j++) {
                    int byteIdx = qsPos + chunk * 32 + j;
                    int qByte = buffer.get(byteIdx) & 0xFF;
                    int q = nibble == 0 ? (qByte & 0x0F) : (qByte >>> 4);
                    out[block * QK_K + group * 32 + j] = scale * q - bias;
                }
            }
            pos += BLOCK_BYTES_Q4_K;
        }
        return out;
    }

    private static void unpackQ4KScales(java.nio.ByteBuffer buffer, int offset, float[] sc, float[] mn) {
        int[] d = new int[4];
        int[] m = new int[4];
        int[] md = new int[4];
        for (int i = 0; i < 4; i++) {
            d[i] = buffer.get(offset + i) & 0xFF;
            m[i] = buffer.get(offset + 4 + i) & 0xFF;
            md[i] = buffer.get(offset + 8 + i) & 0xFF;
        }
        for (int i = 0; i < 4; i++) {
            sc[i] = d[i] & 0x3F;
            sc[i + 4] = (md[i] & 0x0F) | ((d[i] >>> 2) & 0x30);
            mn[i] = m[i] & 0x3F;
            mn[i + 4] = ((md[i] >>> 4) & 0x0F) | ((m[i] >>> 2) & 0x30);
        }
    }

    /** {@code dequantize_row_q6_K} — ggml K-quant super-block (256 elems / 210 bytes). */
    public static float[] dequantizeQ6_K(java.nio.ByteBuffer buffer, int offset, GgmlTensorShape shape) {
        int numElements = (int) shape.numElements();
        if (numElements % QK_K != 0) {
            throw new IllegalArgumentException("Q6_K elements " + numElements + " not divisible by " + QK_K);
        }
        float[] out = new float[numElements];
        int numBlocks = numElements / QK_K;
        int pos = offset;
        for (int block = 0; block < numBlocks; block++) {
            float d = readFp16(buffer, pos + 208);
            int outBase = block * QK_K;
            for (int pass = 0; pass < 2; pass++) {
                int qlOff = pos + pass * 64;
                int qhOff = pos + 128 + pass * 32;
                int scOff = pos + 192 + pass * 8;
                int yOff = outBase + pass * 128;
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int ql0 = buffer.get(qlOff + l) & 0xFF;
                    int ql32 = buffer.get(qlOff + l + 32) & 0xFF;
                    int qhL = buffer.get(qhOff + l) & 0xFF;
                    int q1 = ((ql0 & 0xF) | (((qhL >> 0) & 3) << 4)) - 32;
                    int q2 = ((ql32 & 0xF) | (((qhL >> 2) & 3) << 4)) - 32;
                    int q3 = ((ql0 >> 4) | (((qhL >> 4) & 3) << 4)) - 32;
                    int q4 = ((ql32 >> 4) | (((qhL >> 6) & 3) << 4)) - 32;
                    out[yOff + l] = d * buffer.get(scOff + is) * q1;
                    out[yOff + l + 32] = d * buffer.get(scOff + is + 2) * q2;
                    out[yOff + l + 64] = d * buffer.get(scOff + is + 4) * q3;
                    out[yOff + l + 96] = d * buffer.get(scOff + is + 6) * q4;
                }
            }
            pos += BLOCK_BYTES_Q6_K;
        }
        return out;
    }

    public static float[] dequantizeF16(java.nio.ByteBuffer buffer, int offset, GgmlTensorShape shape) {
        int numElements = (int) shape.numElements();
        float[] out = new float[numElements];
        int pos = offset;
        for (int i = 0; i < numElements; i++) {
            out[i] = readFp16(buffer, pos);
            pos += 2;
        }
        return out;
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
