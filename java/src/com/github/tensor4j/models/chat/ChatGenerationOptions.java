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

/** Sampling hyperparameters (tinygrad llama.py + apps/llm.py defaults). */
public record ChatGenerationOptions(
        ChatGenerationMode mode,
        float temperature,
        float topP,
        int topK,
        int minNewTokens,
        int maxNewTokens,
        long seed,
        int bosId,
        int eosId,
        int eotId,
        float alphaFrequency,
        float alphaPresence,
        boolean gumbelMax,
        int prefillChunkSize,
        ChatSamplingRngMode rngMode) {

    public static final float DEFAULT_TEMPERATURE = 0.7f;
    public static final float DEFAULT_TOP_P = 0.9f;
    public static final int DEFAULT_TOP_K = 40;
    public static final int DEFAULT_MIN_NEW_TOKENS = 2;
    public static final int DEFAULT_MAX_NEW_TOKENS = 128;
    public static final float DEFAULT_ALPHA_FREQUENCY = 0.05f;
    public static final float DEFAULT_ALPHA_PRESENCE = 0.1f;
    public static final int DEFAULT_PREFILL_CHUNK_SIZE = 128;

    public static ChatGenerationOptions quality(ChatTokenizer tokenizer) {
        return quality(tokenizer, ChatSamplingRngMode.SECURE);
    }

    public static ChatGenerationOptions quality(ChatTokenizer tokenizer, ChatSamplingRngMode rngMode) {
        return new ChatGenerationOptions(
                ChatGenerationMode.QUALITY,
                DEFAULT_TEMPERATURE,
                DEFAULT_TOP_P,
                DEFAULT_TOP_K,
                DEFAULT_MIN_NEW_TOKENS,
                DEFAULT_MAX_NEW_TOKENS,
                42L,
                tokenizer.bosId(),
                tokenizer.eosId(),
                tokenizer.eotId(),
                DEFAULT_ALPHA_FREQUENCY,
                DEFAULT_ALPHA_PRESENCE,
                true,
                DEFAULT_PREFILL_CHUNK_SIZE,
                rngMode);
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
                tokenizer.eosId(),
                tokenizer.eotId(),
                0f,
                0f,
                false,
                DEFAULT_PREFILL_CHUNK_SIZE,
                ChatSamplingRngMode.SECURE);
    }

    public static ChatGenerationOptions fromEnvironment(ChatTokenizer tokenizer) {
        ChatGenerationMode mode = ChatGenerationMode.fromString(
                firstNonBlank(System.getenv("TENSOR4J_CHAT_MODE"), System.getProperty("tensor4j.chat.mode")));
        ChatSamplingRngMode rngMode = ChatSamplingRngMode.fromString(
                firstNonBlank(System.getenv("TENSOR4J_CHAT_RNG"), System.getProperty("tensor4j.chat.rng")));
        if (mode == ChatGenerationMode.GREEDY) {
            return new ChatGenerationOptions(
                    ChatGenerationMode.GREEDY,
                    0f,
                    1f,
                    0,
                    0,
                    envInt("TENSOR4J_CHAT_MAX_TOKENS", DEFAULT_MAX_NEW_TOKENS),
                    envLong("TENSOR4J_CHAT_SEED", 0L),
                    tokenizer.bosId(),
                    tokenizer.eosId(),
                    tokenizer.eotId(),
                    0f,
                    0f,
                    false,
                    envInt("TENSOR4J_CHAT_PREFILL_CHUNK", DEFAULT_PREFILL_CHUNK_SIZE),
                    rngMode);
        }
        ChatGenerationOptions defaults = quality(tokenizer, rngMode);
        return new ChatGenerationOptions(
                ChatGenerationMode.QUALITY,
                envFloat("TENSOR4J_CHAT_TEMPERATURE", defaults.temperature()),
                envFloat("TENSOR4J_CHAT_TOP_P", defaults.topP()),
                envInt("TENSOR4J_CHAT_TOP_K", defaults.topK()),
                envInt("TENSOR4J_CHAT_MIN_TOKENS", defaults.minNewTokens()),
                envInt("TENSOR4J_CHAT_MAX_TOKENS", defaults.maxNewTokens()),
                envLong("TENSOR4J_CHAT_SEED", defaults.seed()),
                tokenizer.bosId(),
                tokenizer.eosId(),
                tokenizer.eotId(),
                envFloat("TENSOR4J_CHAT_ALPHA_F", defaults.alphaFrequency()),
                envFloat("TENSOR4J_CHAT_ALPHA_P", defaults.alphaPresence()),
                envBool("TENSOR4J_CHAT_GUMBEL", defaults.gumbelMax()),
                envInt("TENSOR4J_CHAT_PREFILL_CHUNK", defaults.prefillChunkSize()),
                rngMode);
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

    private static float envFloat(String key, float defaultValue) {
        String raw = System.getenv(key);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }
        return Float.parseFloat(raw.trim());
    }

    private static int envInt(String key, int defaultValue) {
        String raw = System.getenv(key);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }
        return Integer.parseInt(raw.trim());
    }

    private static long envLong(String key, long defaultValue) {
        String raw = System.getenv(key);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }
        return Long.parseLong(raw.trim());
    }

    private static boolean envBool(String key, boolean defaultValue) {
        String raw = System.getenv(key);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }
        String normalized = raw.trim().toLowerCase();
        return "1".equals(normalized) || "true".equals(normalized) || "yes".equals(normalized);
    }
}
