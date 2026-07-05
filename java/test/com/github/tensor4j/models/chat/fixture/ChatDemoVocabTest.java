/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.chat.fixture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.BpePreType;
import com.github.tensor4j.models.chat.ChatGenerationOptions;
import com.github.tensor4j.models.chat.ChatGenerationResult;
import com.github.tensor4j.models.chat.ChatGenerator;
import com.github.tensor4j.models.chat.ChatHistoryMode;
import com.github.tensor4j.models.chat.ChatModel;
import com.github.tensor4j.models.chat.ChatTemplate;
import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder.ChatDemo;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class ChatDemoVocabTest {

    @AfterEach
    void clearCache() {
        ChatDemoVocab.clearCacheForTests();
    }

    @Test
    void defaultModeIsFull() {
        assertEquals(ChatDemoVocabMode.FULL, ChatDemoVocabMode.fromEnvironment());
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.FULL)
    void fullVocabMatchesLlama32Size() {
        ChatDemoVocab.InferenceVocab vocab = ChatDemoVocab.load(ChatDemoVocabMode.FULL);
        assertEquals(ChatDemoVocabMode.FULL, vocab.mode());
        assertEquals(ChatDemo.FULL_VOCAB, vocab.vocabSize());
        assertEquals(ChatDemo.FULL_VOCAB, vocab.fullVocabSize());
        assertEquals("llama-bpe", vocab.pre());
        assertFalse(vocab.ignoreMerges());
        assertTrue(vocab.merges().length > 100_000);
        assertEquals("<|begin_of_text|>", vocab.tokens()[vocab.bosTokenId()]);
        assertEquals("<|eot_id|>", vocab.tokens()[vocab.eosTokenId()]);
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void prunedVocabIsSmallerSlice() {
        ChatDemoVocab.InferenceVocab vocab = ChatDemoVocab.load(ChatDemoVocabMode.PRUNED);
        assertEquals(ChatDemoVocabMode.PRUNED, vocab.mode());
        assertTrue(vocab.vocabSize() >= ChatDemo.PRUNED_MIN_VOCAB);
        assertTrue(vocab.vocabSize() < ChatDemo.FULL_VOCAB);
        assertEquals(ChatDemo.FULL_VOCAB, vocab.fullVocabSize());
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.FULL)
    void fullOpenFixtureEncodesHelloAndGenerates() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildOpenChatDemoModel());
        assertEquals(BpePreType.LLAMA3, model.tokenizer().preType());
        assertEquals(ChatDemo.FULL_VOCAB, model.config().nVocab());
        assertTrue(model.tokenizer().encode("Hello").length > 0);

        ChatGenerationResult result = new ChatGenerator(
                        model, ChatGenerationOptions.greedy(model.tokenizer(), 16), ChatHistoryMode.LEGACY)
                .generate("Hello", ChatTemplate.PLAIN);
        assertTrue(result.text().length() > 1, result::text);
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void prunedOpenFixtureEncodesHello() {
        ChatModel model = ChatModel.fromGguf(
                MiniChatGgufBuilder.buildOpenChatDemoModel(ChatDemoVocab.load(ChatDemoVocabMode.PRUNED)));
        assertTrue(model.config().nVocab() < ChatDemo.FULL_VOCAB);
        assertTrue(model.tokenizer().encode("Hello").length > 0);
    }
}
