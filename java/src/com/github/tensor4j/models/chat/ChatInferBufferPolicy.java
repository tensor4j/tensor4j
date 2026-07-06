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

import java.io.PrintStream;
import java.util.Locale;

/**
 * Config-guarded isolation for prompt / logits / session buffers between inference calls.
 *
 * <p>Defaults favour defensive copies — disable individual flags only when hunting aliasing bugs.
 */
public final class ChatInferBufferPolicy {

    private ChatInferBufferPolicy() {}

    /**
     * Clone {@code int[]} prompt slices before {@link ChatModel#forward(int[])} (default {@code true}).
     *
     * <p>Env: {@code TENSOR4J_CHAT_COPY_PROMPT_TOKENS}.
     */
    public static boolean copyPromptTokens() {
        return parseBool("TENSOR4J_CHAT_COPY_PROMPT_TOKENS", true);
    }

    /**
     * Clone logits returned from {@link ChatModel#forward(int[])} before sampling or storage (default
     * {@code true}).
     *
     * <p>Env: {@code TENSOR4J_CHAT_CLONE_FORWARD_LOGITS}. Legacy alias: {@code TENSOR4J_CHAT_CLONE_LOGITS}.
     */
    public static boolean cloneForwardLogits() {
        String legacy = System.getenv("TENSOR4J_CHAT_CLONE_LOGITS");
        if (legacy != null && !legacy.isBlank()) {
            return Boolean.parseBoolean(legacy.trim());
        }
        return parseBool("TENSOR4J_CHAT_CLONE_FORWARD_LOGITS", true);
    }

    /** @see #cloneForwardLogits() */
    public static boolean cloneLogitsBeforeSample() {
        return cloneForwardLogits();
    }

    /**
     * Clone session token arrays when committing turn state (default {@code true}).
     *
     * <p>Env: {@code TENSOR4J_CHAT_DEFENSIVE_SESSION_COPY}.
     */
    public static boolean defensiveSessionCopy() {
        return parseBool("TENSOR4J_CHAT_DEFENSIVE_SESSION_COPY", true);
    }

    /**
     * Log {@link System#identityHashCode(Object)} for prompt/logit arrays when debug is on (default
     * {@code false}).
     *
     * <p>Env: {@code TENSOR4J_CHAT_LOG_BUFFER_IDENTITY}.
     */
    public static boolean logBufferIdentity() {
        return parseBool("TENSOR4J_CHAT_LOG_BUFFER_IDENTITY", false);
    }

    /**
     * Log rendered ChatML prompt text immediately before tokenization (default {@code false}).
     *
     * <p>Env: {@code TENSOR4J_CHAT_LOG_PROMPT_TEXT}.
     */
    public static boolean logPromptTextBeforeTokenize() {
        return parseBool("TENSOR4J_CHAT_LOG_PROMPT_TEXT", false);
    }

    public static int[] isolatePromptTokens(int[] tokens) {
        return copyPromptTokens() ? tokens.clone() : tokens;
    }

    public static float[] isolateForwardLogits(float[] logits) {
        return cloneForwardLogits() ? logits.clone() : logits;
    }

    public static int[] isolateSessionTokens(int[] tokens) {
        return defensiveSessionCopy() ? tokens.clone() : tokens;
    }

    public static void logIdentity(PrintStream out, String label, Object buffer) {
        if (!logBufferIdentity() || !ChatTokenDebugLog.enabled() || buffer == null) {
            return;
        }
        out.printf(
                Locale.ROOT,
                "  buffer_identity %s: hash=%d class=%s%n",
                label,
                System.identityHashCode(buffer),
                buffer.getClass().getSimpleName());
    }

    static boolean parseBool(String key, boolean defaultValue) {
        String raw = System.getenv(key);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }
        return Boolean.parseBoolean(raw.trim());
    }
}
