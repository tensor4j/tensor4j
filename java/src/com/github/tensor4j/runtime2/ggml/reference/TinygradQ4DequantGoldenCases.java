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

/** Hardcoded Q4_0 dequant cases aligned with tinygrad {@code test_gguf.py} block layout. */
public final class TinygradQ4DequantGoldenCases {

    private TinygradQ4DequantGoldenCases() {
    }

    public static TinygradQ4DequantGoldenCase[] all() {
        return new TinygradQ4DequantGoldenCase[] {
                singleBlockScaleTwo(),
                singleBlockAllCenterNibbles(),
                matrix32x2RoundTrip(),
        };
    }

    /** scale=2.0, nibbles 8..15 in both halves → values 0,2,..,14 repeated. */
    private static TinygradQ4DequantGoldenCase singleBlockScaleTwo() {
        byte[] block = new byte[GgmlQuant.BLOCK_BYTES_Q4_0];
        GgmlFp16.writeLittleEndian(block, 0, GgmlFp16.fromFloat32(2.0f));
        for (int j = 0; j < GgmlQuant.QK4_0 / 2; j++) {
            int nibble = 8 + (j % 8);
            block[2 + j] = (byte) (nibble | (nibble << 4));
        }
        float[] expected = new float[GgmlQuant.QK4_0];
        for (int i = 0; i < expected.length; i++) {
            expected[i] = (i % 8) * 2.0f;
        }
        return new TinygradQ4DequantGoldenCase(
                "single_block_scale_2",
                new int[] {GgmlQuant.QK4_0},
                block,
                expected,
                1e-5f);
    }

    /** All nibbles 8 → dequant 0 regardless of scale. */
    private static TinygradQ4DequantGoldenCase singleBlockAllCenterNibbles() {
        byte[] block = new byte[GgmlQuant.BLOCK_BYTES_Q4_0];
        GgmlFp16.writeLittleEndian(block, 0, GgmlFp16.fromFloat32(0.5f));
        for (int j = 2; j < block.length; j++) {
            block[j] = (byte) 0x88;
        }
        float[] expected = new float[GgmlQuant.QK4_0];
        return new TinygradQ4DequantGoldenCase(
                "single_block_center_nibbles",
                new int[] {GgmlQuant.QK4_0},
                block,
                expected,
                1e-6f);
    }

    /** 64-element matrix via reference quantizer (smoke shape for token_embd-style weights). */
    private static TinygradQ4DequantGoldenCase matrix32x2RoundTrip() {
        float[] source = new float[GgmlQuant.QK4_0 * 2];
        for (int i = 0; i < source.length; i++) {
            source[i] = (float) Math.sin(i * 0.31);
        }
        byte[] blocks = GgmlQuant.quantizeRowQ4_0Reference(source);
        return new TinygradQ4DequantGoldenCase(
                "matrix_32x2_roundtrip",
                new int[] {32, 2},
                blocks,
                source,
                0.2f);
    }

    public static GgmlTensorShape toShape(int[] dims) {
        return switch (dims.length) {
            case 1 -> GgmlTensorShape.of(dims[0]);
            case 2 -> GgmlTensorShape.of(dims[0], dims[1]);
            case 3 -> GgmlTensorShape.of(dims[0], dims[1], dims[2]);
            case 4 -> GgmlTensorShape.of(dims[0], dims[1], dims[2], dims[3]);
            default -> throw new IllegalArgumentException("rank " + dims.length);
        };
    }
}
