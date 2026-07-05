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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.runtime.gguf.MmappedGgufFile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import org.junit.jupiter.api.Test;

/** llama.cpp-style delta tokenization parity (fixture + optional live GGUF). */
class LlamaCppTokenParityTest {

    @Test
    void llama3FullPromptEqualsLegacyTemplate() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel().header());
        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);
        List<ChatMessage> messages = List.of(new ChatMessage("user", "hello"));
        assertArrayEquals(
                ChatTemplate.LLAMA3.encodePromptForGeneration(tokenizer, "hello"),
                applier.tokenIds(tokenizer, messages, true));
    }

    @Test
    void llama3TurnTwoTokenDeltaMatchesSuffix() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel().header());
        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);
        List<ChatMessage> afterOne = List.of(
                new ChatMessage("user", "hello"),
                new ChatMessage("assistant", "hi"));
        int prev = applier.tokenCountAfterAssistantTurn(tokenizer, afterOne);
        List<ChatMessage> turnTwo = List.of(
                new ChatMessage("user", "hello"),
                new ChatMessage("assistant", "hi"),
                new ChatMessage("user", "again"));
        int[] full = applier.tokenIds(tokenizer, turnTwo, true);
        assertArrayEquals(
                java.util.Arrays.copyOfRange(full, prev, full.length),
                applier.tokenDeltaSince(tokenizer, turnTwo, true, prev));
    }

    @Test
    void liveGgufFirstTurnMatchesLegacyWhenPathSet() throws Exception {
        String ggufEnv = System.getenv("TENSOR4J_GGUF_PATH");
        assumeTrue(ggufEnv != null && !ggufEnv.isBlank());
        Path ggufPath = Paths.get(ggufEnv.trim());
        assumeTrue(Files.isRegularFile(ggufPath));

        try (MmappedGgufFile mapped = MmappedGgufFile.open(ggufPath)) {
            ChatTokenizer tokenizer = ChatTokenizer.fromGguf(mapped.header());
            assumeTrue(
                    tokenizer.preType() == BpePreType.LLAMA3
                            || tokenizer.preType() == BpePreType.QWEN2
                            || tokenizer.preType() == BpePreType.QWEN35);
            ChatTemplate template = ChatTemplate.fromTokenizer(tokenizer);
            LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);
            String user = "What is the opposite of dark?";
            int[] legacy = template.encodePromptForGeneration(tokenizer, user);
            int[] llama = applier.tokenIds(tokenizer, List.of(new ChatMessage("user", user)), true);
            assertArrayEquals(legacy, llama);
        }
    }
}
