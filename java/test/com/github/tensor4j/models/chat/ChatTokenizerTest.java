/*
 * Copyright 2026 IcedTea-Web Maintainers
 * Copyright 2026 Tensor4j Maintainers
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.chat;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import org.junit.jupiter.api.Test;

class ChatTokenizerTest {

    @Test
    void encodesSingleCharTokens() {
        ChatTokenizer tok = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildIdentityModel().header());
        assertArrayEquals(new int[] {1}, tok.encode("a"));
        assertArrayEquals(new int[] {1, 2}, tok.encode("ab"));
    }

    @Test
    void decodesRoundTrip() {
        ChatTokenizer tok = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildIdentityModel().header());
        assertEquals("ab", tok.decode(tok.encode("ab")));
    }

    @Test
    void readsBosEosFromMetadata() {
        ChatTokenizer tok = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildIdentityModel().header());
        assertEquals(0, tok.bosId());
        assertEquals(3, tok.eosId());
        assertEquals(MiniChatGgufBuilder.N_VOCAB, tok.vocabSize());
    }

    @Test
    void encodeWithBosPrependsToken() {
        ChatTokenizer tok = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildIdentityModel().header());
        assertArrayEquals(new int[] {0, 1}, tok.encodeWithBos("a"));
    }
}
