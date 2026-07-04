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

import java.util.Locale;

/**
 * Llama 3.2 tokenizer fixture scope for chat-demo GGUF builds.
 *
 * <p>{@link #FULL} — entire {@code tokenizer.ggml.*} from tinygrad {@code apps/llm.py} llama3.2:1b
 * (default). {@link #PRUNED} — smaller checked-in/dev slice for fast chain-routing smoke tests.
 */
public enum ChatDemoVocabMode {

    FULL,
    PRUNED;

    /** Env {@code TENSOR4J_CHAT_VOCAB} / property {@code tensor4j.chat.vocab}: {@code full} (default) or {@code pruned}. */
    public static ChatDemoVocabMode fromEnvironment() {
        String raw = firstNonBlank(
                System.getenv("TENSOR4J_CHAT_VOCAB"),
                System.getProperty("tensor4j.chat.vocab"));
        if (raw == null || raw.isBlank()) {
            return FULL;
        }
        String normalized = raw.trim().toLowerCase(Locale.ROOT);
        if ("pruned".equals(normalized) || "slice".equals(normalized) || "mini".equals(normalized)) {
            return PRUNED;
        }
        return FULL;
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
