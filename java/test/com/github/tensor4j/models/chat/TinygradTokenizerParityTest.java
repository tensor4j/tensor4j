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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.models.chat.reference.TinygradTokenizerGoldenCase;
import com.github.tensor4j.models.chat.reference.TinygradTokenizerGoldenCases;
import com.github.tensor4j.models.chat.reference.TinygradTokenizerReference;
import com.github.tensor4j.runtime.gguf.GgufFile;
import org.junit.jupiter.api.Test;

/** tinygrad {@code SimpleTokenizer} + {@link ChatTemplate} parity on fixture vocabs. */
class TinygradTokenizerParityTest {

    @Test
    void goldenEncodeCasesMatchChatTokenizer() {
        for (TinygradTokenizerGoldenCase golden : TinygradTokenizerGoldenCases.all()) {
            if (!"encode".equals(golden.kind())) {
                continue;
            }
            ChatTokenizer tokenizer = ChatTokenizer.fromGguf(TinygradTokenizerGoldenCases.llama3BpeFixture().header());
            assertArrayEquals(
                    golden.expectedIds(),
                    TinygradTokenizerGoldenCases.expectedFor(tokenizer, golden),
                    golden.name());
        }
    }

    @Test
    void decodeRoundTripMatchesGolden() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(TinygradTokenizerGoldenCases.llama3BpeFixture().header());
        TinygradTokenizerGoldenCase golden = TinygradTokenizerGoldenCases.all()[1];
        assertEquals(golden.expectedDecode(), tokenizer.decode(golden.expectedIds()));
    }

    @Test
    void llama3RoleHeaderEncodesFromTemplateFixture() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(TinygradTokenizerGoldenCases.llama3TemplateFixture().header());
        int[] role = TinygradTokenizerReference.role(tokenizer, "user");
        assertTrue(role.length > 0);
        assertArrayEquals(role, ChatTemplate.LLAMA3.encodeRole(tokenizer, "user"));
    }

    @Test
    void llama3UserTurnMatchesRolePlusMessage() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(TinygradTokenizerGoldenCases.llama3TemplateFixture().header());
        int[] viaTemplate = ChatTemplate.LLAMA3.encodeUser(tokenizer, "Hello");
        int[] viaRef = concat(TinygradTokenizerReference.role(tokenizer, "user"), tokenizer.encode("Hello"));
        assertArrayEquals(viaRef, viaTemplate);
    }

    @Test
    void endTurnMatchesReference() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(TinygradTokenizerGoldenCases.llama3TemplateFixture().header());
        assertArrayEquals(
                TinygradTokenizerReference.endTurn(tokenizer, tokenizer.eosId()),
                ChatTemplate.LLAMA3.encodeEndTurn(tokenizer));
    }

    @Test
    void qwen2ChatMlBoundariesMatchTinygradGoldenCases() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(TinygradTokenizerGoldenCases.qwen2TemplateFixture().header());
        for (var golden : TinygradTokenizerGoldenCases.qwen2Cases()) {
            assertArrayEquals(
                    golden.expectedIds(),
                    TinygradTokenizerGoldenCases.expectedFor(tokenizer, golden),
                    golden.name());
        }
    }

    private static int[] concat(int[] a, int[] b) {
        int[] out = new int[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }
}
