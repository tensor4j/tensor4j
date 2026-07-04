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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.tensor4j.support.TokenizerGgufFixtures;
import org.junit.jupiter.api.Test;

class ChatTokenizerPreTest {

    @Test
    void llama3UsesWholeWordLookupBeforeMerges() {
        String[] tokens = {"<s>", "Hello", "!", "</s>"};
        ChatTokenizer tok = ChatTokenizer.fromGguf(TokenizerGgufFixtures.header("llama3", tokens));
        assertEquals(BpePreType.LLAMA3, tok.preType());
        assertArrayEquals(new int[] {1}, tok.encode("Hello"));
    }

    @Test
    void gemma4EncodesNewlineSeparatedChunks() {
        String[] tokens = {"a", "b", "\n", "ab"};
        ChatTokenizer tok = ChatTokenizer.fromGguf(TokenizerGgufFixtures.header("gemma4", tokens));
        assertEquals(BpePreType.GEMMA4, tok.preType());
        assertArrayEquals(new int[] {0, 2, 1}, tok.encode("a\nb"));
    }

    @Test
    void qwen2AndLlama3DifferOnLongNumbers() {
        String[] tokens = {"1", "2", "3", "4", "5", "123", "45", "12345", "<s>", "</s>"};
        ChatTokenizer llama3 = ChatTokenizer.fromGguf(TokenizerGgufFixtures.header("llama3", tokens));
        ChatTokenizer qwen2 = ChatTokenizer.fromGguf(TokenizerGgufFixtures.header("qwen2", tokens));
        assertArrayEquals(new int[] {0, 1, 2, 3, 4}, qwen2.encode("12345"));
        assertArrayEquals(new int[] {5, 6}, llama3.encode("12345"));
    }

    @Test
    void gpt2PreUsesGpt2Profile() {
        String[] tokens = {"a", "b", "ab", "<s>", "</s>"};
        ChatTokenizer tok = ChatTokenizer.fromGguf(TokenizerGgufFixtures.header("gpt-2", tokens));
        assertEquals(BpePreType.GPT2, tok.preType());
        assertArrayEquals(new int[] {0, 1}, tok.encode("ab"));
    }

    @Test
    void explicitIgnoreMergesOverridesPreDefault() {
        String[] tokens = {"<s>", "Hello", "</s>"};
        ChatTokenizer forced = ChatTokenizer.fromGguf(
                TokenizerGgufFixtures.header("qwen2", tokens, new String[0], true));
        assertArrayEquals(new int[] {1}, forced.encode("Hello"));
    }
}
