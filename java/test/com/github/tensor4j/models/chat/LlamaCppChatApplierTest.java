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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import java.util.List;
import org.junit.jupiter.api.Test;

class LlamaCppChatApplierTest {

    @Test
    void llama3TokenIdsMatchLegacyTemplateOnFirstTurn() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel().header());
        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);
        List<ChatMessage> messages = List.of(new ChatMessage("user", "hello"));

        int[] llamaIds = applier.tokenIds(tokenizer, messages, true);
        int[] legacyIds = ChatTemplate.LLAMA3.encodePromptForGeneration(tokenizer, "hello");
        assertArrayEquals(legacyIds, llamaIds);
    }

    @Test
    void llama3TurnOneMatchesSimpleChatShape() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel().header());
        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);
        List<ChatMessage> messages = List.of(new ChatMessage("user", "hello"));

        String full = applier.apply(messages, true);
        assertFalse(full.isEmpty());
        assertEquals(full.length(), applier.deltaSince(messages, true, 0).length());
    }

    @Test
    void llama3TurnTwoTokenDeltaExcludesCommittedPrefix() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel().header());
        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);
        List<ChatMessage> afterTurnOne = List.of(
                new ChatMessage("user", "hello"),
                new ChatMessage("assistant", "hi there"));
        int prevTokens = applier.tokenCountAfterAssistantTurn(tokenizer, afterTurnOne);

        List<ChatMessage> turnTwo = List.of(
                new ChatMessage("user", "hello"),
                new ChatMessage("assistant", "hi there"),
                new ChatMessage("user", "second"));
        int[] delta = applier.tokenDeltaSince(tokenizer, turnTwo, true, prevTokens);
        int[] full = applier.tokenIds(tokenizer, turnTwo, true);
        assertArrayEquals(java.util.Arrays.copyOfRange(full, prevTokens, full.length), delta);
        assertTrue(delta.length > 0);
    }

    @Test
    void qwen2TurnOneOpensAssistantSlot() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel().header());
        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);
        int[] ids = applier.tokenIds(tokenizer, List.of(new ChatMessage("user", "hello")), true);
        int[] legacy = ChatTemplate.QWEN2.encodePromptForGeneration(tokenizer, "hello");
        assertArrayEquals(legacy, ids);
    }
}
