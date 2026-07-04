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

import com.github.tensor4j.runtime.ggml.GgmlTensorShape;
import com.github.tensor4j.runtime.ggml.GgmlType;
import java.util.List;
import org.junit.jupiter.api.Test;

/** GGUF metadata parse round-trip (gguf.h layout). */
class GgufReaderTest {

    @Test
    void roundTripMinimalHeader() {
        GgufKvEntry arch = new GgufKvEntry("general.architecture", GgufType.STRING, "llama");
        GgufKvEntry alignment = new GgufKvEntry(GgufConstants.KEY_GENERAL_ALIGNMENT, GgufType.UINT32, 32);
        GgufTensorInfo weight = GgufWriter.tensor("token_embd.weight",
                GgmlType.Q4_0, GgmlTensorShape.of(32, 2), 0);
        byte[] blob = GgufWriter.writeMetadata(GgufConstants.VERSION, List.of(arch, alignment), List.of(weight));
        GgufHeader header = GgufReader.readMetadata(blob);
        assertEquals(GgufConstants.VERSION, header.version());
        assertEquals(32, header.alignment());
        assertEquals("llama", header.findKv("general.architecture").value());
        GgufTensorInfo parsed = header.findTensor("token_embd.weight");
        assertNotNull(parsed);
        assertEquals(GgmlType.Q4_0, parsed.type());
        assertEquals(36, parsed.sizeBytes());
        assertEquals(0, parsed.offsetBytes());
        assertEquals(GgmlTensorShape.of(32, 2), parsed.shape());
    }

    @Test
    void f32TensorMetadata() {
        GgufTensorInfo info = GgufWriter.tensor("norm.weight", GgmlType.F32, GgmlTensorShape.of(128), 512);
        byte[] blob = GgufWriter.writeMetadata(GgufConstants.VERSION, List.of(), List.of(info));
        GgufTensorInfo parsed = GgufReader.readMetadata(blob).findTensor("norm.weight");
        assertEquals(512, parsed.offsetBytes());
        assertEquals(512, parsed.sizeBytes());
        assertEquals(128, parsed.shape().ne(0));
    }

    @Test
    void dataOffsetIsAligned() {
        GgufKvEntry kv = new GgufKvEntry("general.name", GgufType.STRING, "test");
        GgufTensorInfo t = GgufWriter.tensor("w", GgmlType.F32, GgmlTensorShape.of(8), 0);
        byte[] blob = GgufWriter.writeMetadata(GgufConstants.VERSION, List.of(kv), List.of(t));
        GgufHeader header = GgufReader.readMetadata(blob);
        assertEquals(0, header.dataOffset() % header.alignment());
    }
}
