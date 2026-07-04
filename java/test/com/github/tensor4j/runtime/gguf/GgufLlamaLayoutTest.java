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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import org.junit.jupiter.api.Test;

/** tinygrad {@code from_gguf} Q/K row permute parity. */
class GgufLlamaLayoutTest {

    @Test
    void permuteSwapsInterleavedHeadPairs() {
        int nHeads = 2;
        int headDim = 4;
        int rows = nHeads * headDim;
        int cols = 2;
        float[] data = new float[rows * cols];
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                data[row * cols + col] = row * 10f + col;
            }
        }
        float before = data[1 * cols];
        GgufLlamaLayout.permuteQkInterleaved(data, rows, cols, nHeads);
        assertNotEquals(before, data[1 * cols]);
        assertEquals(0f, data[0 * cols], 1e-6f);
        assertEquals(20f, data[1 * cols], 1e-6f);
        assertEquals(10f, data[2 * cols], 1e-6f);
        assertEquals(30f, data[3 * cols], 1e-6f);
    }

    @Test
    void reverseGgufDimsIsIdentityOnFlatBuffer() {
        float[] gguf = {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f};
        float[] reversed = GgufLlamaLayout.reverseGgufDims(gguf, 4, 2);
        for (int i = 0; i < gguf.length; i++) {
            assertEquals(gguf[i], reversed[i], 1e-6f);
        }
    }
}
