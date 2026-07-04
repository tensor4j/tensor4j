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

import com.github.tensor4j.runtime.ggml.GgmlTensorShape;
import com.github.tensor4j.runtime.ggml.GgmlType;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import org.junit.jupiter.api.Test;

/** GGUF data-section offset rules (gguf.cpp). */
class GgufWeightLayoutTest {

    @Test
    void sequentialOffsetsWithPadding() {
        int alignment = 32;
        GgufTensorPayload a = payload("a", GgmlType.F32, GgmlTensorShape.of(8), f32Bytes(1, 2, 3, 4, 5, 6, 7, 8));
        GgufTensorPayload b = payload("b", GgmlType.Q4_0, GgmlTensorShape.of(32, 2),
                new byte[36]);
        List<GgufTensorInfo> plan = GgufWeightLayout.plan(List.of(a, b), alignment);
        assertEquals(0, plan.get(0).offsetBytes());
        assertEquals(32, plan.get(1).offsetBytes());
        assertEquals(96, GgufWeightLayout.totalDataBytes(plan, alignment));
    }

    @Test
    void emptyPayloads() {
        assertEquals(0, GgufWeightLayout.totalDataBytes(List.of(), 32));
    }

    private static GgufTensorPayload payload(String name, GgmlType type, GgmlTensorShape shape, byte[] data) {
        return new GgufTensorPayload(name, type, shape, data);
    }

    private static byte[] f32Bytes(float... values) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float value : values) {
            buffer.putFloat(value);
        }
        return buffer.array();
    }
}
