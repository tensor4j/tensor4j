/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime2.ggml.reference;

import com.github.tensor4j.runtime.ggml.GgmlFp16;
import com.github.tensor4j.runtime.ggml.GgmlQuant;
import com.github.tensor4j.runtime.ggml.GgmlTensorShape;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Port of tinygrad {@code ggml_data_to_tensor} for Q4_0 (ggml type id 2).
 *
 * <p>tinygrad reshapes quant bytes to {@code (-1, 18)} blocks, then:
 * {@code (q_to_uint8(blocks[:,2:], 4).cast(int8) - 8) * blocks[:,:2].bitcast(float16)}.
 */
public final class TinygradQ4DequantReference {

    private TinygradQ4DequantReference() {
    }

    public static float[] dequantize(byte[] quantBytes, GgmlTensorShape shape) {
        ByteBuffer buffer = ByteBuffer.wrap(quantBytes).order(ByteOrder.LITTLE_ENDIAN);
        return dequantize(buffer, 0, shape);
    }

    public static float[] dequantize(ByteBuffer buffer, int offset, GgmlTensorShape shape) {
        int numElements = (int) shape.numElements();
        if (numElements % GgmlQuant.QK4_0 != 0) {
            throw new IllegalArgumentException("numElements " + numElements + " not divisible by "
                    + GgmlQuant.QK4_0);
        }
        int numBlocks = numElements / GgmlQuant.QK4_0;
        int expectedBytes = numBlocks * GgmlQuant.BLOCK_BYTES_Q4_0;
        if (buffer.remaining() - offset < expectedBytes) {
            throw new IllegalArgumentException("need " + expectedBytes + " quant bytes, have "
                    + (buffer.remaining() - offset));
        }
        float[] out = new float[numElements];
        int pos = offset;
        for (int block = 0; block < numBlocks; block++) {
            float scale = readFp16(buffer, pos);
            pos += 2;
            for (int j = 0; j < GgmlQuant.QK4_0 / 2; j++) {
                int packed = buffer.get(pos++) & 0xFF;
                int q0 = (packed & 0x0F) - 8;
                int q1 = (packed >>> 4) - 8;
                int base = block * GgmlQuant.QK4_0 + j;
                out[base] = q0 * scale;
                out[base + GgmlQuant.QK4_0 / 2] = q1 * scale;
            }
        }
        return out;
    }

    /** Block count and byte length for a ggml Q4_0 tensor (tinygrad slice sizing). */
    public static int quantBytes(GgmlTensorShape shape) {
        long elements = shape.numElements();
        if (elements % GgmlQuant.QK4_0 != 0) {
            throw new IllegalArgumentException("elements " + elements + " not divisible by " + GgmlQuant.QK4_0);
        }
        return (int) (elements / GgmlQuant.QK4_0 * GgmlQuant.BLOCK_BYTES_Q4_0);
    }

    private static float readFp16(ByteBuffer buffer, int offset) {
        int bits = (buffer.get(offset) & 0xFF) | ((buffer.get(offset + 1) & 0xFF) << 8);
        return GgmlFp16.toFloat32(bits);
    }
}
