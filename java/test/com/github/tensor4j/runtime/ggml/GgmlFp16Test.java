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

import org.junit.jupiter.api.Test;

class GgmlFp16Test {

    @Test
    void roundTripCommonValues() {
        assertEquals(0.0f, GgmlFp16.toFloat32(GgmlFp16.fromFloat32(0.0f)), 0.0f);
        assertEquals(1.0f, GgmlFp16.toFloat32(GgmlFp16.fromFloat32(1.0f)), 1e-3f);
        assertEquals(-2.5f, GgmlFp16.toFloat32(GgmlFp16.fromFloat32(-2.5f)), 1e-2f);
    }

    @Test
    void knownHalfOne() {
        byte[] block = new byte[] {(byte) 0x00, (byte) 0x3C};
        assertEquals(1.0f, GgmlFp16.toFloat32(block, 0), 1e-3f);
    }
}
