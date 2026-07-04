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

/** Sampling hyperparameters (llama-style chat defaults). */
public record ChatGenerationOptions(
        ChatGenerationMode mode,
        float temperature,
        float topP,
        int topK,
        int minNewTokens,
        int maxNewTokens,
        long seed,
        int bosId,
        int eosId) {

    public static final float DEFAULT_TEMPERATURE = 0.7f;
    public static final float DEFAULT_TOP_P = 0.9f;
    public static final int DEFAULT_TOP_K = 40;
    public static final int DEFAULT_MIN_NEW_TOKENS = 2;
    public static final int DEFAULT_MAX_NEW_TOKENS = 32;

    public static ChatGenerationOptions quality(ChatTokenizer tokenizer) {
        return new ChatGenerationOptions(
                ChatGenerationMode.QUALITY,
                DEFAULT_TEMPERATURE,
                DEFAULT_TOP_P,
                DEFAULT_TOP_K,
                DEFAULT_MIN_NEW_TOKENS,
                DEFAULT_MAX_NEW_TOKENS,
                42L,
                tokenizer.bosId(),
                tokenizer.eosId());
    }

    public static ChatGenerationOptions greedy(ChatTokenizer tokenizer, int maxNewTokens) {
        return new ChatGenerationOptions(
                ChatGenerationMode.GREEDY,
                0f,
                1f,
                0,
                0,
                maxNewTokens,
                0L,
                tokenizer.bosId(),
                tokenizer.eosId());
    }

    public static ChatGenerationOptions fromEnvironment(ChatTokenizer tokenizer) {
        ChatGenerationMode mode = ChatGenerationMode.fromString(
                firstNonBlank(System.getenv("TENSOR4J_CHAT_MODE"), System.getProperty("tensor4j.chat.mode")));
        if (mode == ChatGenerationMode.GREEDY) {
            return greedy(tokenizer, DEFAULT_MAX_NEW_TOKENS);
        }
        return quality(tokenizer);
    }

    private static String firstNonBlank(String a, String b) {
        if (a != null && !a.isBlank()) {
            return a;
        }
        if (b != null && !b.isBlank()) {
            return b;
        }
        return null;
    }
}
