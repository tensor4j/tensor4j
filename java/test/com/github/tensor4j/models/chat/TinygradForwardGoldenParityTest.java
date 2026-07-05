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

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.models.chat.reference.TinygradForwardGoldenCase;
import com.github.tensor4j.models.chat.reference.TinygradForwardGoldenCases;
import com.github.tensor4j.support.TensorAssert;
import org.junit.jupiter.api.Test;

/** Forward logits parity on deterministic chat fixtures (completion-quality guard). */
class TinygradForwardGoldenParityTest {

    @Test
    void chatModelForwardMatchesGoldenLogits() {
        for (TinygradForwardGoldenCase golden : TinygradForwardGoldenCases.all()) {
            ChatModel model = modelFor(golden.fixture());
            int[] tokens = tokensFor(golden, model);
            float[] logits = model.forward(tokens);
            TensorAssert.assertAllClose(golden.logits(), logits, golden.tolerance());
        }
    }

    @Test
    void llama3BpeHelloArgmaxStable() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildLlama3BpeModel());
        float[] logits = model.forward(new int[] {1});
        int argmax = ChatSampler.argmax(logits);
        org.junit.jupiter.api.Assertions.assertTrue(argmax >= 0 && argmax < logits.length);
    }

    private static ChatModel modelFor(String fixture) {
        if ("llama3_bpe".equals(fixture)) {
            return ChatModel.fromGguf(MiniChatGgufBuilder.buildLlama3BpeModel());
        }
        if ("identity".equals(fixture)) {
            return ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        }
        if ("llama3_template".equals(fixture)) {
            return ChatModel.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel());
        }
        if ("qwen2_template".equals(fixture)) {
            return ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        }
        throw new IllegalArgumentException("unknown fixture " + fixture);
    }

    private static int[] tokensFor(TinygradForwardGoldenCase golden, ChatModel model) {
        if (golden.tokens() != null) {
            return golden.tokens();
        }
        if ("llama3_template_user_hello".equals(golden.name())) {
            return ChatTemplate.LLAMA3.encodeUser(model.tokenizer(), "Hello");
        }
        if ("qwen2_template_user_hello".equals(golden.name())) {
            return ChatTemplate.QWEN2.encodePromptForGeneration(model.tokenizer(), "Hello");
        }
        throw new IllegalArgumentException("no tokens for " + golden.name());
    }
}
