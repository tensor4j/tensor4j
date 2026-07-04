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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.runtime.gguf.MmappedGgufFile;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

/**
 * Optional smoke test against a real on-disk GGUF model.
 * Set {@code TENSOR4J_GGUF_PATH} to a llama-architecture {@code .gguf} file.
 */
@EnabledIfEnvironmentVariable(named = "TENSOR4J_GGUF_PATH", matches = ".+")
class ExternalGgufSmokeTest {

    @Test
    void mmapTokenizeForwardAndSample() throws Exception {
        Path path = Paths.get(System.getenv("TENSOR4J_GGUF_PATH"));
        try (MmappedGgufFile mapped = MmappedGgufFile.open(path)) {
            ChatTokenizer tokenizer = ChatTokenizer.fromGguf(mapped.header());
            ChatModel model = ChatModel.fromGguf(mapped);

            int[] prompt = tokenizer.encode("Hello");
            assertFalse(prompt.length == 0);

            float[] logits = model.forward(prompt);
            assertEquals(model.config().nVocab(), logits.length);

            int next = ChatSampler.argmax(logits);
            assertTrue(next >= 0 && next < logits.length);

            String piece = tokenizer.decode(new int[] {next});
            assertNotNull(piece);
        }
    }
}
