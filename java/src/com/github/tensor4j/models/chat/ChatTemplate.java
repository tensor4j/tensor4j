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
    };

    public abstract int[] encodeUser(ChatTokenizer tokenizer, String message);

    /** BOS / {@code prefix()} for chat models (tinygrad {@code SimpleTokenizer.prefix}). */
    public int[] encodePrefix(ChatTokenizer tokenizer) {
        if (this == LLAMA3 && tokenizer.bosId() >= 0) {
            return new int[] {tokenizer.bosId()};
        }
        return new int[0];
    }

    /** User role + message + {@code end_turn} (llama3 only). */
    public int[] encodeUserTurn(ChatTokenizer tokenizer, String message) {
        if (this == LLAMA3) {
            return concat(encodeUser(tokenizer, message), encodeEndTurn(tokenizer));
        }
        return encodeUser(tokenizer, message);
    }

    /** Opens the assistant generation slot after a user turn. */
    public int[] encodeAssistantPrime(ChatTokenizer tokenizer) {
        if (this == LLAMA3) {
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
        if (this == LLAMA3) {
            return TinygradTokenizerReference.endTurn(tokenizer, tokenizer.eosId());
        }
        return new int[0];
    }

    public static ChatTemplate fromEnvironment() {
        String raw = System.getenv("TENSOR4J_CHAT_TEMPLATE");
        if (raw == null || raw.isBlank()) {
            return PLAIN;
        }
        String normalized = raw.trim().toLowerCase(java.util.Locale.ROOT);
        if ("llama3".equals(normalized) || "llama-3".equals(normalized)) {
            return LLAMA3;
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
