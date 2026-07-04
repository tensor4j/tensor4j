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

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.ChatGenerationOptions;
import com.github.tensor4j.models.chat.ChatSamplingRngMode;
import com.github.tensor4j.models.chat.ChatGenerationResult;
import com.github.tensor4j.models.chat.ChatGenerator;
import com.github.tensor4j.models.chat.ChatModel;
import com.github.tensor4j.models.chat.ChatTemplate;
import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;

/** Decode loop helpers for chat-demo integration tests. */
final class ChatSession {

    static final int DEFAULT_MAX_NEW_TOKENS = ChatGenerationOptions.DEFAULT_MAX_NEW_TOKENS;

    private ChatSession() {
    }

    static ChatModel loadMiniModel() {
        return ChatModel.fromGguf(MiniChatGgufBuilder.buildChatDemoModel());
    }

    /** Seeded open-ended level-12 fixture (tinygrad real-GGUF path, not token chain). */
    static ChatModel loadOpenModel() {
        return ChatModel.fromGguf(MiniChatGgufBuilder.buildOpenChatDemoModel());
    }

    static ChatTemplate templateForDemo() {
        return ChatTemplate.fromEnvironment();
    }

    /** Quality sampling defaults (SecureRandom) — completions vary run-to-run. */
    static ChatGenerationOptions optionsFor(ChatModel model) {
        return ChatGenerationOptions.fromEnvironment(model.tokenizer());
    }

    /** Seeded legacy RNG for sampler reproducibility unit tests only. */
    static ChatGenerationOptions optionsForSeededDemo(ChatModel model) {
        return ChatGenerationOptions.quality(model.tokenizer(), ChatSamplingRngMode.LEGACY);
    }

    /** Model produced non-trivial decoded text (sampling variance allowed). */
    static void assertRealCompletion(String text) {
        assertTrue(text.length() > 1, () -> "expected completion length > 1, got: " + text);
    }

    static GenerationResult generate(ChatModel model, String prompt, ChatGenerationOptions options) {
        return generate(model, prompt, options, ChatTemplate.PLAIN);
    }

    static GenerationResult generate(
            ChatModel model, String prompt, ChatGenerationOptions options, ChatTemplate template) {
        ChatGenerator generator = new ChatGenerator(model, options);
        ChatGenerationResult result = generator.generate(prompt, template);
        return new GenerationResult(
                result.text(),
                result.tokenCount(),
                result.mode(),
                result.prefixReuseTokens());
    }

    static GenerationResult greedyGenerate(ChatModel model, String prompt, int maxNewTokens) {
        ChatGenerationOptions options = ChatGenerationOptions.greedy(model.tokenizer(), maxNewTokens);
        return generate(model, prompt, options);
    }

    record GenerationResult(String text, int tokenCount, String mode, int prefixReuseTokens) {
    }
}
