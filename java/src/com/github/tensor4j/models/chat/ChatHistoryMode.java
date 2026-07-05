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

import java.util.Locale;

/**
 * Multi-turn conversation history strategy.
 *
 * <ul>
 *   <li>{@link #LLAMA} — llama.cpp {@code simple-chat}: structured messages, Jinja-style
 *       template from GGUF, delta tokenization each turn (default).
 *   <li>{@link #LEGACY} — tensor4j/tinygrad {@code apps/llm.py}: full token-id replay with KV
 *       prefix reuse.
 * </ul>
 */
public enum ChatHistoryMode {

    LLAMA,
    LEGACY;

    /** {@code TENSOR4J_CHAT_HISTORY_MODE} — {@code llama} (default) or {@code legacy}. */
    public static ChatHistoryMode fromEnvironment() {
        String raw = System.getenv("TENSOR4J_CHAT_HISTORY_MODE");
        if (raw == null || raw.isBlank()) {
            return InferCompatMode.fromEnvironment().defaultHistoryMode();
        }
        return parseName(raw);
    }

    public static ChatHistoryMode parseName(String raw) {
        String normalized = raw.trim().toLowerCase(Locale.ROOT);
        if ("legacy".equals(normalized) || "tensor4j".equals(normalized) || "tinygrad".equals(normalized)) {
            return LEGACY;
        }
        if ("llama".equals(normalized) || "llama.cpp".equals(normalized) || "llamacpp".equals(normalized)) {
            return LLAMA;
        }
        return LLAMA;
    }
}
