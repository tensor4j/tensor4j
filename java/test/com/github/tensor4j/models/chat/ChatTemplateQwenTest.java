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

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.runtime.gguf.GgufFile;
import org.junit.jupiter.api.Test;

/** Qwen2 ChatML prompt structure (tinygrad {@code SimpleTokenizer} qwen branch). */
class ChatTemplateQwenTest {

    @Test
    void qwenTurnEndsWithImEndAndNewline() {
        GgufFile file = MiniChatGgufBuilder.buildQwen2TemplateModel();
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(file.header());
        assertEquals(BpePreType.QWEN2, tokenizer.preType());
        assertTrue(tokenizer.eosId() != tokenizer.eotId(), "fixture must split eos vs im_end like real Qwen GGUF");

        int[] userTurn = ChatTemplate.QWEN2.encodeUserTurn(tokenizer, "Hello");
        int imEnd = tokenizer.eotId();
        assertTrue(lastIndexOf(userTurn, imEnd) >= 0, "user turn must include im_end");
        assertTrue(lastIndexOf(userTurn, tokenizer.tokenIdForText("<|im_start|>")) == 0);

        int[] prompt = ChatTemplate.QWEN2.encodePromptForGeneration(tokenizer, "Hello");
        assertEquals(0, ChatTemplate.QWEN2.encodePrefix(tokenizer).length, "qwen chat has no BOS prefix");
        assertTrue(prompt.length > userTurn.length);
    }

    @Test
    void qwenEndTurnAfterImEndIsNewlineOnly() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel().header());
        int imEnd = tokenizer.eotId();
        int[] suffix = ChatTemplate.QWEN2.encodeEndTurnAfter(tokenizer, new int[] {imEnd});
        int newlineId = tokenizer.tokenIdForText("\n");
        assertEquals(1, suffix.length);
        assertEquals(newlineId, suffix[0]);
    }

    private static int lastIndexOf(int[] array, int value) {
        for (int i = array.length - 1; i >= 0; i--) {
            if (array[i] == value) {
                return i;
            }
        }
        return -1;
    }

    @Test
    void fromTokenizerSelectsQwen2() {
        GgufFile file = MiniChatGgufBuilder.buildQwen2TemplateModel();
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(file.header());
        assertEquals(ChatTemplate.QWEN2, ChatTemplate.fromTokenizer(tokenizer));
    }
}
