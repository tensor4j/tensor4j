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
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import org.junit.jupiter.api.Test;

class InferCompatModeTest {

    @Test
    void parseNames() {
        assertEquals(InferCompatMode.TINYGRAD, InferCompatMode.parseName("tinygrad"));
        assertEquals(InferCompatMode.TINYGRAD, InferCompatMode.parseName("tensor4j"));
        assertEquals(InferCompatMode.LLAMA_CPP, InferCompatMode.parseName("llama"));
        assertEquals(InferCompatMode.LLAMA_CPP, InferCompatMode.parseName("llama.cpp"));
    }

    @Test
    void tinygradModeUsesLegacyHistoryWhenEnvUnset() {
        assertEquals(ChatHistoryMode.LEGACY, InferCompatMode.TINYGRAD.defaultHistoryMode());
    }

    @Test
    void llamaCppModeUsesDeltaHistoryWhenEnvUnset() {
        assertEquals(ChatHistoryMode.LLAMA, InferCompatMode.LLAMA_CPP.defaultHistoryMode());
    }

    @Test
    void sampledTokenIdsGatedByCompatMode() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        ChatGenerator llamaCompat = new ChatGenerator(
                model, ChatGenerationOptions.greedy(model.tokenizer(), 2), ChatHistoryMode.LLAMA);
        ChatGenerationResult result = llamaCompat.continueConversation("Hi", ChatTemplate.QWEN2);
        ChatMessage last = llamaCompat.messages().get(llamaCompat.messages().size() - 1);
        if (InferCompatMode.fromEnvironment().useSampledAssistantTokenIds()) {
            assertTrue(last.generatedTokenIds() != null);
            assertEquals(result.forwardedTokenIds().length, last.generatedTokenIds().length);
        } else {
            assertTrue(last.generatedTokenIds() == null || last.generatedTokenIds().length == 0);
        }
    }

    @Test
    void eotMaskDuringMinTokensFollowsCompatMode() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildLlama3BpeModel().header());
        ChatGenerationOptions options = new ChatGenerationOptions(
                ChatGenerationMode.QUALITY,
                0.7f,
                1f,
                0,
                2,
                8,
                0L,
                tokenizer.bosId(),
                tokenizer.eosId(),
                tokenizer.eotId(),
                0f,
                0f,
                false,
                128,
                ChatSamplingRngMode.LEGACY);
        float[] logits = new float[tokenizer.vocabSize()];
        for (int i = 0; i < logits.length; i++) {
            logits[i] = 1f;
        }
        int picked = ChatSampler.sample(logits, options, 0, new ChatSamplerState(logits.length), new java.util.Random(0));
        if (InferCompatMode.fromEnvironment().maskEndTokensDuringMinNewTokens()) {
            assertFalse(picked == tokenizer.eosId() || picked == tokenizer.eotId() || picked == tokenizer.bosId());
        }
    }
}
