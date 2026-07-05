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

    /** Qwen2 / Qwen2.5 ChatML ({@code <|im_start|>role\\n ... \\n}). */
    QWEN2 {
        @Override
        public int[] encodeUser(ChatTokenizer tokenizer, String message) {
            return concat(TinygradTokenizerReference.role(tokenizer, "user"), tokenizer.encode(message));
        }
    };

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

    /**
     * Full prefill for one-shot generation: prefix + user turn + assistant prime
     * (tinygrad chat server message loop).
     */
    public int[] encodePromptForGeneration(ChatTokenizer tokenizer, String message) {
        return concat(
                encodePrefix(tokenizer),
                concat(encodeUserTurn(tokenizer, message), encodeAssistantPrime(tokenizer)));
    }

    /** tinygrad {@code SimpleTokenizer.role(role)}. */
    public int[] encodeRole(ChatTokenizer tokenizer, String role) {
        return TinygradTokenizerReference.role(tokenizer, role);
    }

    /** tinygrad {@code SimpleTokenizer.end_turn(eos_id)}. */
    public int[] encodeEndTurn(ChatTokenizer tokenizer) {
        if (usesStructuredTurns()) {
            return TinygradTokenizerReference.endTurn(tokenizer, tokenizer.eosId());
        }
        return new int[0];
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
