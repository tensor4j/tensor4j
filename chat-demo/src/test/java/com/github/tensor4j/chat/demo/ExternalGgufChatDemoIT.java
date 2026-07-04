/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.chat.demo;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.ChatModel;
import com.github.tensor4j.models.chat.ChatTemplate;
import com.github.tensor4j.runtime.gguf.MmappedGgufFile;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

/**
 * Optional demo against a real on-disk GGUF ({@code TENSOR4J_GGUF_PATH}).
 * Run via {@code mvn verify} with env set.
 */
@EnabledIfEnvironmentVariable(named = "TENSOR4J_GGUF_PATH", matches = ".+")
class ExternalGgufChatDemoIT {

    @Test
    void externalGgufTokenizesAndGenerates() throws Exception {
        Path path = Paths.get(System.getenv("TENSOR4J_GGUF_PATH"));
        ChatDemoReporter.banner("external GGUF mmap chat");
        try (MmappedGgufFile mapped = MmappedGgufFile.open(path)) {
            ChatModel model = ChatModel.fromGguf(mapped);
            ChatDemoReporter.modelInfo(model, path.toAbsolutePath().toString());

            String prompt = "Hello";
            ChatTemplate template = ChatSession.templateForDemo();
            ChatDemoReporter.tokenize(model.tokenizer(), prompt);
            float[] logits = model.forward(template.encodeUser(model.tokenizer(), prompt));
            assertFalse(Float.isNaN(logits[0]));
            assertTrue(logits.length == model.config().nVocab());

            ChatSession.GenerationResult result = ChatSession.generate(
                    model, prompt, ChatSession.optionsFor(model), template);
            ChatDemoReporter.generation(
                    prompt, result.text(), result.tokenCount(), result.mode(), result.prefixReuseTokens());
            ChatSession.assertRealCompletion(result.text());
        }
    }
}
