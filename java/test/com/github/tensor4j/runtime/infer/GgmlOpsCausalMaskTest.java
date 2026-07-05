/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.infer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Causal mask semantics — llama-graph / tinygrad prefill mask ({@code triu(start_pos+1)}). */
class GgmlOpsCausalMaskTest {

    @Test
    void queryAtGlobalPosCanAttendThroughSameIndex() {
        // pastKv=4, batch=2 -> global positions 4 and 5; K columns 0..6 after append
        float[] raw = {
                1, 1, 1, 1, 1, 0, 0,
                1, 1, 1, 1, 1, 1, 0,
        };
        InferTensor scores = InferTensor.of(raw, 2, 7);
        float[] masked = GgmlOps.applyCausalMask(scores, 4).data();

        assertEquals(1f, masked[4], 1e-6f, "row0 may attend key at global pos 4");
        assertEquals(Float.NEGATIVE_INFINITY, masked[5], 1e-6f, "row0 must not attend future key 5");
        assertEquals(1f, masked[12], 1e-6f, "row1 may attend key at global pos 5");
        assertEquals(Float.NEGATIVE_INFINITY, masked[13], 1e-6f, "row1 must not attend future key 6");
    }

    @Test
    void singleTokenDecodeUsesPastKvAsGlobalPosition() {
        float[] raw = {1, 1, 1, 1, 1};
        InferTensor scores = InferTensor.of(raw, 1, 5);
        float[] masked = GgmlOps.applyCausalMask(scores, 4).data();
        assertEquals(1f, masked[4], 1e-6f);
        for (int c = 0; c < 4; c++) {
            assertEquals(1f, masked[c], 1e-6f, "decode at pos 4 may attend all prior keys");
        }
    }

    @Test
    void prefillFromZeroMasksUpperTriangle() {
        float[] raw = {1, 0, 0, 1, 1, 0, 1, 1, 1};
        InferTensor scores = InferTensor.of(raw, 3, 3);
        float[] masked = GgmlOps.applyCausalMask(scores, 0).data();
        assertEquals(1f, masked[0]);
        assertEquals(Float.NEGATIVE_INFINITY, masked[1]);
        assertEquals(Float.NEGATIVE_INFINITY, masked[2]);
        assertEquals(1f, masked[3]);
        assertEquals(1f, masked[4]);
        assertEquals(Float.NEGATIVE_INFINITY, masked[5]);
        assertTrue(masked[8] > 0f);
    }
}
