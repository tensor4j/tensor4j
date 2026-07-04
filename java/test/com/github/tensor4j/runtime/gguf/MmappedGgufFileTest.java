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

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.support.TensorAssert;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class MmappedGgufFileTest {

    @TempDir
    Path tempDir;

    @Test
    void mmapMatchesHeapLoad() throws Exception {
        GgufFile heap = MiniChatGgufBuilder.buildIdentityModel();
        Path path = tempDir.resolve("mini.gguf");
        Files.write(path, heap.bytes());

        try (MmappedGgufFile mapped = MmappedGgufFile.open(path)) {
            assertEquals(heap.header().dataOffset(), mapped.header().dataOffset());
            assertArrayEquals(heap.tensorBytes("token_embd.weight"), mapped.tensorBytes("token_embd.weight"));
            TensorAssert.assertAllClose(
                    GgufWeightLoader.loadMatrix(heap, "blk.0.attn_k.weight").data(),
                    GgufWeightLoader.loadMatrix(mapped, "blk.0.attn_k.weight").data(),
                    1e-6f);
        }
    }
}
