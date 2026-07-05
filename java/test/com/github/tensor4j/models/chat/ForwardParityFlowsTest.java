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
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.runtime.infer.InferTensor;
import org.junit.jupiter.api.Test;

/**
 * Documents forward parity routing: shared {@link ChatModel} path, architecture-specific weight
 * layout, and {@link InferCompatMode} gating for llama.cpp-only chat fixes.
 */
class ForwardParityFlowsTest {

    @Test
    void llamaAndQwenShareChatModelForwardStack() {
        ChatModel llama = ChatModel.fromGguf(MiniChatGgufBuilder.buildLlama3BpeModel());
        ChatModel qwen = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        assertTrue(llama.forward(new int[] {1}).length > 0);
        assertTrue(qwen.forward(new int[] {1}).length > 0);
    }

    @Test
    void qwenFixtureUsesQwen2ArchitectureForWeightLayout() {
        ChatConfig config = ChatConfig.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel().header());
        assertEquals("qwen2", config.architecture());
        assertTrue(config.isQwen2Family());
    }

    @Test
    void llamaFixtureUsesLlamaArchitectureForQkPermute() {
        ChatConfig config = ChatConfig.fromGguf(MiniChatGgufBuilder.buildLlama3BpeModel().header());
        assertEquals("llama", config.architecture());
        assertFalse(config.isQwen2Family());
    }

    @Test
    void qwenAndLlamaIdentityHelloLogitsDifferWhenLayoutsDiverge() {
        ChatModel llama = ChatModel.fromGguf(MiniChatGgufBuilder.buildLlama3BpeModel());
        ChatModel qwen = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        float[] llamaLogits = llama.forward(new int[] {1});
        float[] qwenLogits = qwen.forward(new int[] {1});
        assertNotEquals(llamaLogits.length, qwenLogits.length);
    }

    @Test
    void qwenLoadsQueryMatrixWithoutLlamaQkPermute() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        InferTensor wq = model.weights().layer(0).wq().tensor();
        assertEquals(model.config().nEmbd(), wq.rows());
        assertEquals(model.config().nEmbd(), wq.cols());
    }

    @Test
    void tinygradCompatDisablesLlamaCppChatFixes() {
        InferCompatMode mode = InferCompatMode.TINYGRAD;
        assertFalse(mode.useSampledAssistantTokenIds());
        assertFalse(mode.maskEndTokensDuringMinNewTokens());
        assertEquals(ChatHistoryMode.LEGACY, mode.defaultHistoryMode());
    }

    @Test
    void llamaCppCompatEnablesDeltaKvFixes() {
        InferCompatMode mode = InferCompatMode.LLAMA_CPP;
        assertTrue(mode.useSampledAssistantTokenIds());
        assertTrue(mode.maskEndTokensDuringMinNewTokens());
        assertEquals(ChatHistoryMode.LLAMA, mode.defaultHistoryMode());
    }
}
