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

class TokenizerWhitespacePreTest {

    @Test
    void whitespacePreSplitsOnWhitespaceBeforeBpe() {
        String[] tokens = {"<s>", "hello", "world", "</s>"};
        ChatTokenizer tok = ChatTokenizer.fromGguf(
                TokenizerGgufFixtures.header("whitespace", tokens));
        assertEquals(BpePreType.WHITESPACE, tok.preType());
        assertArrayEquals(new int[] {1}, tok.encode("hello"));
        assertArrayEquals(new int[] {1, 2}, tok.encode("hello world"));
    }

    @Test
    void mapsRemainingPreStrings() {
        assertEquals(BpePreType.WHITESPACE, BpePreType.fromPre("whitespace"));
        assertEquals(BpePreType.GPT2, BpePreType.fromPre("jina-v1-en"));
        assertEquals(BpePreType.GPT2, BpePreType.fromPre("jina-v2-code"));
        assertEquals(BpePreType.GPT2, BpePreType.fromPre("roberta-bpe"));
    }
}
