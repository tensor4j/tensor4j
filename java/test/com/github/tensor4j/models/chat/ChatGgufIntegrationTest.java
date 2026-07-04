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

import com.github.tensor4j.runtime.gguf.GgufFile;
import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import org.junit.jupiter.api.Test;

/** In-process GGUF encode → forward → sample smoke (always runs in CI). */
class ChatGgufIntegrationTest {

    @Test
    void miniLlama3GgufEncodesForwardAndSamples() {
        GgufFile file = MiniChatGgufBuilder.buildLlama3BpeModel();
        ChatConfig config = ChatConfig.fromGguf(file.header());
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(file.header());
        ChatWeights weights = ChatGgufLoader.loadViews(file, config);
        ChatModel model = ChatModel.withRingCache(config, tokenizer, weights);

        assertEquals(BpePreType.LLAMA3, tokenizer.preType());
        assertArrayEquals(new int[] {1}, tokenizer.encode("Hello"));

        float[] logits = model.forward(tokenizer.encode("Hello"));
        assertEquals(config.nVocab(), logits.length);

        int next = ChatSampler.argmax(logits);
        assertTrue(next >= 0 && next < config.nVocab());

        model.resetCache();
        float[] viaText = model.forwardText("Hello");
        assertEquals(logits.length, viaText.length);
        assertFalse(Float.isNaN(viaText[0]));
    }
}
