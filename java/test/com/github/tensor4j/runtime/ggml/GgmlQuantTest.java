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

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.tensor4j.support.TensorAssert;
import org.junit.jupiter.api.Test;

/** Q4_0 dequant (ggml-quants.c dequantize_row_q4_0). */
class GgmlQuantTest {

    @Test
    void dequantSingleBlockAllZeros() {
        byte[] blocks = new byte[GgmlQuant.BLOCK_BYTES_Q4_0];
        GgmlFp16.writeLittleEndian(blocks, 0, GgmlFp16.fromFloat32(0.5f));
        for (int j = 2; j < blocks.length; j++) {
            blocks[j] = (byte) 0x88;
        }
        float[] out = new float[GgmlQuant.QK4_0];
        GgmlQuant.dequantizeRowQ4_0(blocks, 0, out, 0, GgmlQuant.QK4_0);
        for (float value : out) {
            assertEquals(0.0f, value, 1e-6f);
        }
    }

    @Test
    void quantReferenceRoundTrip() {
        float[] source = new float[GgmlQuant.QK4_0];
        for (int i = 0; i < source.length; i++) {
            source[i] = (i - 16) * 0.25f;
        }
        byte[] blocks = GgmlQuant.quantizeRowQ4_0Reference(source);
        float[] restored = new float[GgmlQuant.QK4_0];
        GgmlQuant.dequantizeRowQ4_0(blocks, 0, restored, 0, GgmlQuant.QK4_0);
        TensorAssert.assertAllClose(source, restored, 0.25f);
    }

    @Test
    void dequantTwoBlocks() {
        float[] source = new float[GgmlQuant.QK4_0 * 2];
        for (int i = 0; i < source.length; i++) {
            source[i] = (float) Math.sin(i * 0.3);
        }
        byte[] blocks = GgmlQuant.quantizeRowQ4_0Reference(source);
        float[] restored = GgmlQuant.dequantizeQ4_0(blocks, GgmlTensorShape.of(GgmlQuant.QK4_0, 2));
        TensorAssert.assertAllClose(source, restored, 0.2f);
    }
}
