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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/** ggml tensor layout math (ggml.h / gguf.cpp stride rules). */
class GgmlLayoutTest {

    @Test
    void f32MatrixNumBytes() {
        GgmlTensorShape shape = GgmlTensorShape.of(4, 8);
        assertEquals(128, GgmlLayout.numBytes(GgmlType.F32, shape));
    }

    @Test
    void f32ByteStrides() {
        GgmlTensorShape shape = GgmlTensorShape.of(4, 8);
        assertArrayEquals(new long[] {4, 16, 128, 128}, GgmlLayout.byteStrides(GgmlType.F32, shape));
    }

    @Test
    void q4RowNumBytes() {
        GgmlTensorShape shape = GgmlTensorShape.of(32, 2);
        assertEquals(36, GgmlLayout.numBytes(GgmlType.Q4_0, shape));
    }

    @Test
    void q4ByteStrides() {
        GgmlTensorShape shape = GgmlTensorShape.of(64, 3);
        assertArrayEquals(new long[] {18, 36, 108, 108}, GgmlLayout.byteStrides(GgmlType.Q4_0, shape));
    }

    @Test
    void padAlignment() {
        assertEquals(32, GgmlLayout.pad(17, 32));
        assertEquals(64, GgmlLayout.pad(64, 32));
    }

    @Test
    void numElementsRank() {
        GgmlTensorShape shape = GgmlTensorShape.of(2, 3, 4, 1);
        assertEquals(24, shape.numElements());
        assertEquals(3, shape.rank());
    }
}
