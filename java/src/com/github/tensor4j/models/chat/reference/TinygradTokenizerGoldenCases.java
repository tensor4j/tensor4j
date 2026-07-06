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

    public static GgufFile qwen2TemplateFixture() {
        return MiniChatGgufBuilder.buildQwen2TemplateModel();
    }

    /** tinygrad {@code SimpleTokenizer} qwen2 branch on {@link #qwen2TemplateFixture()}. */
    public static TinygradTokenizerGoldenCase[] qwen2Cases() {
        return new TinygradTokenizerGoldenCase[] {
                qwenRole("user", new int[] {2, 3, 6}),
                qwenRole("system", new int[] {2, 5, 6}),
                qwenRole("assistant", new int[] {2, 4, 6}),
                qwenEndTurn(new int[] {7, 6}),
                qwenUserTurnHello(new int[] {2, 3, 6, 1, 7, 6}),
                qwenSystemTurnDefault(new int[] {2, 5, 6, 7, 6}),
                qwenPromptTurn1Hello(new int[] {2, 5, 6, 7, 6, 2, 3, 6, 1, 7, 6, 2, 4, 6}),
        };
    }

    private static TinygradTokenizerGoldenCase qwenSystemTurnDefault(int[] ids) {
        return new TinygradTokenizerGoldenCase(
                "qwen2_system_turn_default", "template_system_turn", ChatTemplate.QWEN2_DEFAULT_SYSTEM, null, ids, null);
    }

    private static TinygradTokenizerGoldenCase qwenPromptTurn1Hello(int[] ids) {
        return new TinygradTokenizerGoldenCase(
                "qwen2_prompt_turn1_hello", "template_prompt_turn1", "Hello", null, ids, null);
    }

    private static TinygradTokenizerGoldenCase qwenRole(String role, int[] ids) {
        return new TinygradTokenizerGoldenCase("qwen2_role_" + role, "role", null, role, ids, null);
    }

    private static TinygradTokenizerGoldenCase qwenEndTurn(int[] ids) {
        return new TinygradTokenizerGoldenCase("qwen2_end_turn", "end_turn", null, null, ids, null);
    }

    private static TinygradTokenizerGoldenCase qwenUserTurnHello(int[] ids) {
        return new TinygradTokenizerGoldenCase("qwen2_user_turn_hello", "template_user_turn", "Hello", "user", ids, null);
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
            case "template_user_turn" -> ChatTemplate.QWEN2.encodeUserTurn(tokenizer, golden.text());
            case "template_system_turn" ->
                    ChatTemplate.QWEN2.encodeSystemTurn(tokenizer, ChatTemplate.QWEN2_DEFAULT_SYSTEM);
            case "template_prompt_turn1" ->
                    ChatTemplate.QWEN2.encodePromptForGeneration(tokenizer, golden.text());
            case "end_turn" -> TinygradTokenizerReference.endTurn(tokenizer, tokenizer.endTurnId());
            default -> throw new IllegalArgumentException("unknown kind " + golden.kind());
        };
    }
}
