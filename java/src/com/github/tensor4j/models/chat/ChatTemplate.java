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

import com.github.tensor4j.models.chat.reference.TinygradTokenizerReference;

/** Prompt formatting before tokenization (tinygrad {@code SimpleTokenizer.role}). */
public enum ChatTemplate {

    /** Encode user text as-is — mini fixtures and smoke tests. */
    PLAIN {
        @Override
        public int[] encodeUser(ChatTokenizer tokenizer, String message) {
            return tokenizer.encode(message);
        }
    },

    /** Llama 3 instruct header wrapper for real chat models. */
    LLAMA3 {
        @Override
        public int[] encodeUser(ChatTokenizer tokenizer, String message) {
            return concat(TinygradTokenizerReference.role(tokenizer, "user"), tokenizer.encode(message));
        }
    },

    /** Qwen2 / Qwen2.5 ChatML: {@code <|im_start|>role\\n ... <|im_end|>\\n} per closed turn. */
    QWEN2 {
        @Override
        public int[] encodeUser(ChatTokenizer tokenizer, String message) {
            return concat(TinygradTokenizerReference.role(tokenizer, "user"), tokenizer.encode(message));
        }
    };

    /** llama.cpp / HF ChatML default (use via {@code TENSOR4J_CHAT_SYSTEM_PROMPT=classic}). */
    public static final String QWEN2_CLASSIC_SYSTEM = "You are a helpful assistant.";

    /** @deprecated use {@link #QWEN2_CLASSIC_SYSTEM} or {@link #defaultSystemPromptText()} */
    public static final String QWEN2_DEFAULT_SYSTEM = QWEN2_CLASSIC_SYSTEM;

    /** Default injected system turn — steers multi-turn toward the latest user message. */
    public static final String QWEN2_FOCUS_NEWEST_USER_SYSTEM =
            "You are a helpful assistant and should focus on the newest user prompt at the end only and ignore all earlier user prompts";

    /**
     * Injected Qwen system text when no {@code system} message is supplied.
     *
     * <p>Default ({@code TENSOR4J_CHAT_SYSTEM_PROMPT} unset or {@code focus}): {@link
     * #QWEN2_FOCUS_NEWEST_USER_SYSTEM}. Set {@code classic} for {@link #QWEN2_CLASSIC_SYSTEM}.
     */
    public static String defaultSystemPromptText() {
        return parseSystemPromptMode(System.getenv("TENSOR4J_CHAT_SYSTEM_PROMPT"));
    }

    static String parseSystemPromptMode(String raw) {
        if (raw == null || raw.isBlank()) {
            return QWEN2_FOCUS_NEWEST_USER_SYSTEM;
        }
        String mode = raw.trim().toLowerCase(java.util.Locale.ROOT);
        if ("classic".equals(mode) || "llama".equals(mode) || "default".equals(mode)) {
            return QWEN2_CLASSIC_SYSTEM;
        }
        if ("focus".equals(mode) || "newest".equals(mode) || "debug".equals(mode)) {
            return QWEN2_FOCUS_NEWEST_USER_SYSTEM;
        }
        return QWEN2_FOCUS_NEWEST_USER_SYSTEM;
    }

    /**
     * Whether Qwen chat prepends a default system turn when callers omit a system message.
     *
     * <p>Default {@code true} ({@code TENSOR4J_CHAT_DEFAULT_SYSTEM} unset). Set to {@code false} for raw
     * llama.cpp {@code llama_chat_apply_template} parity (user-first, no injected system turn).
     */
    public static boolean defaultSystemTurnEnabled() {
        return parseDefaultSystemTurnEnabled(System.getenv("TENSOR4J_CHAT_DEFAULT_SYSTEM"));
    }

    static boolean parseDefaultSystemTurnEnabled(String raw) {
        if (raw == null || raw.isBlank()) {
            return true;
        }
        return Boolean.parseBoolean(raw.trim());
    }

    public abstract int[] encodeUser(ChatTokenizer tokenizer, String message);

    /** Whether this template wraps user/assistant turns with role markers and end-of-turn tokens. */
    public boolean usesStructuredTurns() {
        return this == LLAMA3 || this == QWEN2;
    }

    /** BOS / {@code prefix()} for chat models (tinygrad {@code SimpleTokenizer.prefix}). */
    public int[] encodePrefix(ChatTokenizer tokenizer) {
        if (this == LLAMA3 && tokenizer.bosId() >= 0) {
            return new int[] {tokenizer.bosId()};
        }
        return new int[0];
    }

    /** User role + message + {@code end_turn}. */
    public int[] encodeUserTurn(ChatTokenizer tokenizer, String message) {
        if (usesStructuredTurns()) {
            return concat(encodeUser(tokenizer, message), encodeEndTurn(tokenizer));
        }
        return encodeUser(tokenizer, message);
    }

    /** Opens the assistant generation slot after a user turn. */
    public int[] encodeAssistantPrime(ChatTokenizer tokenizer) {
        if (usesStructuredTurns()) {
            return encodeRole(tokenizer, "assistant");
        }
        return new int[0];
    }

    /** {@code <|im_start|>system\\n ... <|im_end|>\\n} */
    public int[] encodeSystemTurn(ChatTokenizer tokenizer, String systemText) {
        if (!usesStructuredTurns()) {
            return new int[0];
        }
        return concat(concat(encodeRole(tokenizer, "system"), tokenizer.encode(systemText)), encodeEndTurn(tokenizer));
    }

    /** ChatML expects a leading system turn when {@link #defaultSystemTurnEnabled()}. */
    public int[] encodeDefaultSystemTurnIfMissing(ChatTokenizer tokenizer, boolean hasLeadingSystem) {
        if (this == QWEN2 && defaultSystemTurnEnabled() && !hasLeadingSystem) {
            return encodeSystemTurn(tokenizer, defaultSystemPromptText());
        }
        return new int[0];
    }

    /**
     * Full prefill for one-shot generation: prefix + optional default system + user turn + assistant prime
     * (tinygrad chat server message loop).
     */
    public int[] encodePromptForGeneration(ChatTokenizer tokenizer, String message) {
        return concat(
                encodePrefix(tokenizer),
                concat(
                        encodeDefaultSystemTurnIfMissing(tokenizer, false),
                        concat(encodeUserTurn(tokenizer, message), encodeAssistantPrime(tokenizer))));
    }

    /** tinygrad {@code SimpleTokenizer.role(role)}. */
    public int[] encodeRole(ChatTokenizer tokenizer, String role) {
        return TinygradTokenizerReference.role(tokenizer, role);
    }

    /** ChatML turn close — Qwen uses {@link ChatTokenizer#endTurnId()} ({@code im_end}), not {@code endoftext}. */
    public int[] encodeEndTurn(ChatTokenizer tokenizer) {
        if (usesStructuredTurns()) {
            return TinygradTokenizerReference.endTurn(tokenizer, tokenizer.endTurnId());
        }
        return new int[0];
    }

    /**
     * KV suffix after assistant decode. When llama mode stopped on EOG without forwarding it, append
     * {@code im_end} + {@code \\n}; when forwarded ids already end on {@link ChatTokenizer#endTurnId()},
     * append the trailing {@code \\n} only.
     */
    public int[] encodeEndTurnAfter(ChatTokenizer tokenizer, int[] body) {
        if (!usesStructuredTurns()) {
            return new int[0];
        }
        int endId = tokenizer.endTurnId();
        if (body.length > 0 && body[body.length - 1] == endId) {
            return TinygradTokenizerReference.trailingNewlineAfterEndTurn(tokenizer);
        }
        return TinygradTokenizerReference.endTurn(tokenizer, endId);
    }

    public static ChatTemplate fromEnvironment() {
        String raw = System.getenv("TENSOR4J_CHAT_TEMPLATE");
        if (raw == null || raw.isBlank()) {
            return PLAIN;
        }
        return parseName(raw);
    }

    /** Env override when set; otherwise infer from GGUF tokenizer {@code pre} type. */
    public static ChatTemplate fromEnvironmentOrTokenizer(ChatTokenizer tokenizer) {
        String raw = System.getenv("TENSOR4J_CHAT_TEMPLATE");
        if (raw != null && !raw.isBlank()) {
            return parseName(raw);
        }
        return fromTokenizer(tokenizer);
    }

    public static ChatTemplate fromTokenizer(ChatTokenizer tokenizer) {
        if (tokenizer.preType() == BpePreType.QWEN2 || tokenizer.preType() == BpePreType.QWEN35) {
            return QWEN2;
        }
        if (tokenizer.preType() == BpePreType.LLAMA3) {
            return LLAMA3;
        }
        return PLAIN;
    }

    private static ChatTemplate parseName(String raw) {
        String normalized = raw.trim().toLowerCase(java.util.Locale.ROOT);
        if ("llama3".equals(normalized) || "llama-3".equals(normalized)) {
            return LLAMA3;
        }
        if ("qwen2".equals(normalized) || "qwen".equals(normalized) || "qwen2.5".equals(normalized)
                || "qwen-2.5".equals(normalized)) {
            return QWEN2;
        }
        if ("plain".equals(normalized)) {
            return PLAIN;
        }
        return PLAIN;
    }

    static int[] concat(int[] prefix, int[] suffix) {
        int[] out = new int[prefix.length + suffix.length];
        System.arraycopy(prefix, 0, out, 0, prefix.length);
        System.arraycopy(suffix, 0, out, prefix.length, suffix.length);
        return out;
    }
}
