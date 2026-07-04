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

import com.github.tensor4j.models.chat.ChatTemplate;
import com.github.tensor4j.models.chat.ChatTokenizer;
import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.runtime.gguf.GgufFile;

/** Inline tokenizer goldens aligned with tinygrad {@code SimpleTokenizer} on fixture vocabs. */
public final class TinygradTokenizerGoldenCases {

    private TinygradTokenizerGoldenCases() {
    }

    public static TinygradTokenizerGoldenCase[] all() {
        return new TinygradTokenizerGoldenCase[] {
                encodeHello(),
                decodeAb(),
        };
    }

    public static GgufFile llama3BpeFixture() {
        return MiniChatGgufBuilder.buildLlama3BpeModel();
    }

    public static GgufFile llama3TemplateFixture() {
        return MiniChatGgufBuilder.buildLlama3TemplateModel();
    }

    private static TinygradTokenizerGoldenCase encodeHello() {
        return new TinygradTokenizerGoldenCase(
                "encode_hello_llama3_bpe",
                "encode",
                "Hello",
                null,
                new int[] {1},
                null);
    }

    private static TinygradTokenizerGoldenCase decodeAb() {
        return new TinygradTokenizerGoldenCase(
                "decode_ab_llama3_bpe",
                "decode",
                null,
                null,
                new int[] {2, 3},
                "ab");
    }

    public static int[] expectedFor(ChatTokenizer tokenizer, TinygradTokenizerGoldenCase golden) {
        return switch (golden.kind()) {
            case "encode" -> tokenizer.encode(golden.text());
            case "decode" -> throw new IllegalArgumentException("decode uses expectedDecode");
            case "role" -> TinygradTokenizerReference.role(tokenizer, golden.role());
            case "template_user" -> ChatTemplate.LLAMA3.encodeUser(tokenizer, golden.text());
            case "end_turn" -> TinygradTokenizerReference.endTurn(tokenizer, tokenizer.eosId());
            default -> throw new IllegalArgumentException("unknown kind " + golden.kind());
        };
    }
}
