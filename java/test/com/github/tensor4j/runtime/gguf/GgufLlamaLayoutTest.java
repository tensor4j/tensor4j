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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class GgufLlamaLayoutTest {

    @Test
    void reverseGgufDimsTransposesTwoByThree() {
        float[] ggml = {1, 2, 3, 4, 5, 6};
        float[] out = GgufLlamaLayout.reverseGgufDims(ggml, 2, 3);
        assertEquals(6, out.length);
        assertArrayEquals(new float[] {1, 2, 3, 4, 5, 6}, out, 1e-6f);
    }

    @Test
    void reverseGgufDimsIsInvolutoryOnSquare() {
        float[] ggml = {1, 2, 3, 4};
        float[] once = GgufLlamaLayout.reverseGgufDims(ggml, 2, 2);
        float[] twice = GgufLlamaLayout.reverseGgufDims(once, 2, 2);
        assertArrayEquals(ggml, twice, 1e-6f);
    }
}
