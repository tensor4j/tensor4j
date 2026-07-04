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

import com.github.tensor4j.runtime.ggml.GgmlQuant;
import com.github.tensor4j.runtime.ggml.GgmlTensorShape;
import com.github.tensor4j.runtime.ggml.GgmlType;
import com.github.tensor4j.support.TensorAssert;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Full GGUF file read/write with weight blob. */
class GgufFileTest {

    @Test
    void writeAndReadF32Weights() {
        float[] values = {0.25f, -1.0f, 3.5f, 0.0f};
        byte[] weightBytes = f32Bytes(values);
        GgufTensorPayload payload = new GgufTensorPayload("norm.weight", GgmlType.F32,
                GgmlTensorShape.of(4), weightBytes);
        byte[] file = GgufWriter.writeFile(GgufConstants.VERSION, List.of(), List.of(payload));
        GgufFile parsed = GgufReader.readFile(file);
        assertEquals(parsed.header().dataOffset() + 32, file.length);
        assertArrayEquals(weightBytes, parsed.tensorBytes("norm.weight"));
    }

    @Test
    void writeAndReadMultipleTensors() {
        int alignment = 32;
        GgufKvEntry alignKv = new GgufKvEntry(GgufConstants.KEY_GENERAL_ALIGNMENT, GgufType.UINT32, alignment);
        byte[] f32 = f32Bytes(1, 2, 3, 4, 5, 6, 7, 8);
        float[] qSource = new float[64];
        Arrays.fill(qSource, 0.125f);
        byte[] q4 = GgmlQuant.quantizeRowQ4_0Reference(qSource);
        List<GgufTensorPayload> payloads = List.of(
                new GgufTensorPayload("small.f32", GgmlType.F32, GgmlTensorShape.of(8), f32),
                new GgufTensorPayload("emb.q4", GgmlType.Q4_0, GgmlTensorShape.of(32, 2), q4));
        byte[] file = GgufWriter.writeFile(GgufConstants.VERSION, List.of(alignKv), payloads);
        GgufFile parsed = GgufReader.readFile(file);
        GgufHeader header = parsed.header();
        assertEquals(0, header.dataOffset() % alignment);
        GgufTensorInfo first = header.findTensor("small.f32");
        GgufTensorInfo second = header.findTensor("emb.q4");
        assertEquals(0, first.offsetBytes());
        assertEquals(32, second.offsetBytes());
        assertEquals(header.dataOffset() + first.offsetBytes(),
                indexOf(file, f32));
        assertArrayEquals(f32, parsed.tensorBytes("small.f32"));
        float[] restored = GgmlQuant.dequantizeQ4_0(parsed.tensorBytes("emb.q4"), second.shape());
        TensorAssert.assertAllClose(qSource, restored, 0.05f);
    }

    private static int indexOf(byte[] haystack, byte[] needle) {
        outer:
        for (int i = 0; i <= haystack.length - needle.length; i++) {
            for (int j = 0; j < needle.length; j++) {
                if (haystack[i + j] != needle[j]) {
                    continue outer;
                }
            }
            return i;
        }
        return -1;
    }

    private static byte[] f32Bytes(float... values) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float value : values) {
            buffer.putFloat(value);
        }
        return buffer.array();
    }
}
