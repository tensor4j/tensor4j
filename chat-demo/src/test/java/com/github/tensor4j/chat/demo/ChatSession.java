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

import com.github.tensor4j.models.chat.ChatGenerationOptions;
import com.github.tensor4j.models.chat.ChatModel;
import com.github.tensor4j.models.chat.ChatSampler;
import com.github.tensor4j.models.chat.ChatTokenizer;
import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.runtime.gguf.GgufFile;
import java.util.Random;

/** Decode loop helpers for chat-demo integration tests. */
final class ChatSession {

    static final int DEFAULT_MAX_NEW_TOKENS = ChatGenerationOptions.DEFAULT_MAX_NEW_TOKENS;

    private ChatSession() {
    }

    static ChatModel loadMiniModel() {
        return ChatModel.fromGguf(MiniChatGgufBuilder.buildChatDemoModel());
    }

    static ChatGenerationOptions optionsFor(ChatModel model) {
        return ChatGenerationOptions.fromEnvironment(model.tokenizer());
    }

    static GenerationResult generate(ChatModel model, String prompt, ChatGenerationOptions options) {
        model.resetCache();
        ChatTokenizer tokenizer = model.tokenizer();
        Random rng = new Random(options.seed());
        int[] promptIds = tokenizer.encode(prompt);
        float[] logits = model.forward(promptIds);

        StringBuilder completion = new StringBuilder();
        int generated = 0;
        for (int step = 0; step < options.maxNewTokens(); step++) {
            int next = ChatSampler.sample(logits, options, generated, rng);
            if (next == tokenizer.eosId() && generated >= options.minNewTokens()) {
                break;
            }
            if (next == tokenizer.bosId() || next == tokenizer.eosId()) {
                logits = model.forward(new int[] {next});
                continue;
            }
            String piece = tokenizer.decode(new int[] {next});
            completion.append(piece);
            generated++;
            logits = model.forward(new int[] {next});
        }
        return new GenerationResult(completion.toString(), generated, options.mode().name());
    }

    static GenerationResult greedyGenerate(ChatModel model, String prompt, int maxNewTokens) {
        ChatGenerationOptions options = ChatGenerationOptions.greedy(model.tokenizer(), maxNewTokens);
        return generate(model, prompt, options);
    }

    record GenerationResult(String text, int tokenCount, String mode) {
    }
}
