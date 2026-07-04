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
import static org.junit.jupiter.api.Assertions.assertNotNull;

import com.github.tensor4j.runtime.ggml.GgmlLayout;
import com.github.tensor4j.runtime.ggml.GgmlQuant;
import com.github.tensor4j.runtime.ggml.GgmlTensorShape;
import com.github.tensor4j.runtime.ggml.GgmlType;
import com.github.tensor4j.support.TensorAssert;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Mini GGUF fixture mirroring llama.cpp gguf.cpp layout rules (metadata + aligned data section).
 */
class GgufFixtureTest {

    @Test
    void miniLlamaStyleShard() {
        int alignment = 32;
        GgufKvEntry arch = new GgufKvEntry("general.architecture", GgufType.STRING, "llama");
        GgufKvEntry name = new GgufKvEntry("general.name", GgufType.STRING, "tensor4j-fixture");
        GgufKvEntry alignKv = new GgufKvEntry(GgufConstants.KEY_GENERAL_ALIGNMENT, GgufType.UINT32, alignment);

        float[] embedSource = new float[64];
        for (int i = 0; i < embedSource.length; i++) {
            embedSource[i] = (float) ((i % 17) - 8) * 0.05f;
        }
        byte[] embedQ4 = GgmlQuant.quantizeRowQ4_0Reference(embedSource);

        float[] normValues = new float[128];
        for (int i = 0; i < normValues.length; i++) {
            normValues[i] = 1.0f + i * 0.01f;
        }
        byte[] normF32 = f32Bytes(normValues);

        List<GgufTensorPayload> payloads = List.of(
                new GgufTensorPayload("token_embd.weight", GgmlType.Q4_0, GgmlTensorShape.of(32, 2), embedQ4),
                new GgufTensorPayload("blk.0.attn_norm.weight", GgmlType.F32, GgmlTensorShape.of(128), normF32));

        byte[] fixture = GgufWriter.writeFile(GgufConstants.VERSION, List.of(arch, name, alignKv), payloads);
        GgufFile file = GgufReader.readFile(fixture);
        GgufHeader header = file.header();

        assertEquals("llama", header.findKv("general.architecture").value());
        assertEquals(alignment, header.alignment());
        assertEquals(0, header.dataOffset() % alignment);

        List<GgufTensorInfo> tensors = header.tensors();
        assertEquals(2, tensors.size());
        assertEquals(0, tensors.get(0).offsetBytes());
        long expectedSecondOffset = GgmlLayout.pad(tensors.get(0).sizeBytes(), alignment);
        assertEquals(expectedSecondOffset, tensors.get(1).offsetBytes());

        GgufTensorInfo embed = header.findTensor("token_embd.weight");
        assertNotNull(embed);
        assertEquals(GgmlType.Q4_0, embed.type());
        float[] restoredEmbed = GgmlQuant.dequantizeQ4_0(file.tensorBytes("token_embd.weight"), embed.shape());
        TensorAssert.assertAllClose(embedSource, restoredEmbed, 0.15f);

        GgufTensorInfo norm = header.findTensor("blk.0.attn_norm.weight");
        assertNotNull(norm);
        assertEquals(GgmlType.F32, norm.type());
        float[] restoredNorm = f32Values(file.tensorBytes("blk.0.attn_norm.weight"));
        TensorAssert.assertAllClose(normValues, restoredNorm, 1e-6f);

        long dataEnd = header.dataOffset()
                + GgufWeightLayout.totalDataBytes(tensors, alignment);
        assertEquals(dataEnd, fixture.length);
    }

    private static byte[] f32Bytes(float[] values) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float value : values) {
            buffer.putFloat(value);
        }
        return buffer.array();
    }

    private static float[] f32Values(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[bytes.length / 4];
        for (int i = 0; i < out.length; i++) {
            out[i] = buffer.getFloat();
        }
        return out;
    }
}
