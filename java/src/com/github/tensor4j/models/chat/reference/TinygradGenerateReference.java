/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.chat.reference;

import com.github.tensor4j.models.chat.ChatGenerator;

/**
 * Golden reference for tinygrad {@code apps/llm.py} chat generate loop policy.
 *
 * <p>KV prefix reuse ({@code get_start_pos}), chunked prefill slice lengths, EOS stop, and
 * special-token skip-decode match {@link ChatGenerator}.
 */
public final class TinygradGenerateReference {

    private TinygradGenerateReference() {
    }

    /** tinygrad {@code Transformer.get_start_pos(tokens)} — shared prefix with cached prompt. */
    public static int getStartPos(int[] tokens, int[] cachedTokens) {
        return ChatGenerator.sharedPrefixLength(tokens, cachedTokens);
    }

    /** tinygrad chat server: {@code if next_id == eos_id: break} (with optional min-new guard). */
    public static boolean shouldStop(int nextId, int eosId, int generatedCount, int minNewTokens) {
        return nextId == eosId && generatedCount >= minNewTokens;
    }

    /** BOS/EOS and llama3 {@code <|...|>} control tokens are forwarded without decode output. */
    public static boolean skipDecode(int nextId, int bosId, int eosId) {
        if (nextId == bosId || nextId == eosId) {
            return true;
        }
        return false;
    }

    public static boolean skipDecode(String tokenText, int nextId, int bosId, int eosId) {
        if (skipDecode(nextId, bosId, eosId)) {
            return true;
        }
        return tokenText.startsWith("<|") && tokenText.endsWith("|>");
    }
}
