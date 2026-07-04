/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.runtime.gguf.MmappedGgufFile;
import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** End-to-end chat forward from on-disk GGUF via mmap. */
class RealGgufChatTest {

    @TempDir
    Path tempDir;

    @Test
    void twoLayerMmapForwardAndSample() throws Exception {
        byte[] bytes = MiniChatGgufBuilder.buildTwoLayerModel().bytes();
        Path path = tempDir.resolve("two-layer.gguf");
        Files.write(path, bytes);

        try (MmappedGgufFile mapped = MmappedGgufFile.open(path)) {
            ChatModel model = ChatModel.fromGguf(mapped);
            assertEquals(2, model.config().nLayer());
            float[] logits = model.forward(new int[] {1, 0});
            assertEquals(MiniChatGgufBuilder.N_VOCAB, logits.length);
            int next = model.sample(new int[] {2});
            assertTrue(next >= 0 && next < MiniChatGgufBuilder.N_VOCAB);
        }
    }

    @Test
    void q4MmapMatchesF32Forward() throws Exception {
        byte[] q4Bytes = MiniChatGgufBuilder.buildQ4Model().bytes();
        byte[] f32Bytes = MiniChatGgufBuilder.buildIdentityModel().bytes();
        Path q4Path = tempDir.resolve("q4.gguf");
        Path f32Path = tempDir.resolve("f32.gguf");
        Files.write(q4Path, q4Bytes);
        Files.write(f32Path, f32Bytes);

        float[] q4Logits;
        float[] f32Logits;
        try (MmappedGgufFile q4 = MmappedGgufFile.open(q4Path);
                MmappedGgufFile f32 = MmappedGgufFile.open(f32Path)) {
            q4Logits = ChatModel.fromGguf(q4).forward(new int[] {1});
            f32Logits = ChatModel.fromGguf(f32).forward(new int[] {1});
        }
        assertEquals(f32Logits.length, q4Logits.length);
        assertEquals(ChatSampler.argmax(f32Logits), ChatSampler.argmax(q4Logits));
    }
}
