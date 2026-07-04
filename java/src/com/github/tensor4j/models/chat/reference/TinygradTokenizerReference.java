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

import com.github.tensor4j.models.chat.BpePreType;
import com.github.tensor4j.models.chat.ChatTokenizer;

/**
 * Golden reference for tinygrad {@code apps/llm.py} {@code SimpleTokenizer.role} / {@code end_turn}.
 */
public final class TinygradTokenizerReference {

    private static final String LLAMA3_HEADER_START = "<|" + "start_header_id" + "|>";
    private static final String LLAMA3_HEADER_END = "<|" + "end_header_id" + "|>";

    private TinygradTokenizerReference() {
    }

    public static int[] role(ChatTokenizer tokenizer, String role) {
        if (tokenizer.preType() == BpePreType.QWEN2) {
            return concat(
                    new int[] {tokenizer.tokenIdForText("<|im_start|>")},
                    tokenizer.encode(role + "\n"));
        }
        return concat(
                new int[] {tokenizer.tokenIdForText(LLAMA3_HEADER_START)},
                concat(
                        tokenizer.encode(role),
                        concat(
                                new int[] {tokenizer.tokenIdForText(LLAMA3_HEADER_END)},
                                tokenizer.encodeRoleSuffixNewlines())));
    }

    /** {@code SimpleTokenizer.end_turn(eos_id)}. */
    public static int[] endTurn(ChatTokenizer tokenizer, int eosId) {
        if (tokenizer.preType() == BpePreType.QWEN2) {
            return concat(new int[] {eosId}, tokenizer.encode("\n"));
        }
        return new int[] {eosId};
    }

    private static int[] concat(int[] a, int[] b) {
        int[] out = new int[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }
}
