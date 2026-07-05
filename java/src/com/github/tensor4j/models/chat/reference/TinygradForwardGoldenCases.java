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

/** Forward logits goldens for {@link com.github.tensor4j.models.chat.ChatModel} on deterministic fixtures. */
public final class TinygradForwardGoldenCases {

    private TinygradForwardGoldenCases() {
    }

    public static TinygradForwardGoldenCase[] all() {
        return new TinygradForwardGoldenCase[] {
                llama3BpeHello(),
                identityA(),
                identityAb(),
                llama3TemplateUserHello(),
                qwen2TemplateUserHello(),
        };
    }

    /**
     * {@link com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder#buildLlama3BpeModel()}
     * forward on token {@code Hello} (id 1). Regenerate with {@code ForwardGoldenProbe}.
     */
    private static TinygradForwardGoldenCase llama3BpeHello() {
        return new TinygradForwardGoldenCase(
                "llama3_bpe_hello",
                "llama3_bpe",
                new int[] {1},
                new float[] {0.6625948f, 0.69572455f, 0.72885424f, 0.761984f, 0.7951138f},
                1e-4f);
    }

    /** {@link com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder#buildIdentityModel()} token {@code a} (id 1). */
    private static TinygradForwardGoldenCase identityA() {
        return new TinygradForwardGoldenCase(
                "identity_a",
                "identity",
                new int[] {1},
                new float[] {0.6625948f, 0.69572455f, 0.72885424f, 0.761984f},
                1e-4f);
    }

    /** Identity fixture prefill {@code ab} — last-token logits (tinygrad-style batch prefill). */
    private static TinygradForwardGoldenCase identityAb() {
        return new TinygradForwardGoldenCase(
                "identity_ab",
                "identity",
                new int[] {1, 2},
                new float[] {0.7198997f, 0.7470079f, 0.77411616f, 0.80122435f},
                1e-4f);
    }

    /**
     * {@link com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder#buildLlama3TemplateModel()}
     * + {@link com.github.tensor4j.models.chat.ChatTemplate#LLAMA3} user {@code Hello}.
     */
    private static TinygradForwardGoldenCase llama3TemplateUserHello() {
        return new TinygradForwardGoldenCase(
                "llama3_template_user_hello",
                "llama3_template",
                null,
                new float[] {
                    0.79339117f, 0.8122008f, 0.8310103f, 0.8498197f, 0.86862934f, 0.88743883f,
                    0.90624845f, 0.9250579f, 0.85358745f, 0.8723892f, 0.8911909f
                },
                1e-4f);
    }

    /**
     * {@link com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder#buildQwen2TemplateModel()}
     * + {@link com.github.tensor4j.models.chat.ChatTemplate#QWEN2} user {@code Hello}.
     */
    private static TinygradForwardGoldenCase qwen2TemplateUserHello() {
        return new TinygradForwardGoldenCase(
                "qwen2_template_user_hello",
                "qwen2_template",
                null,
                new float[] {0.75902104f, 0.7815475f, 0.804074f, 0.82660043f, 0.84912694f, 0.8716534f, 0.89417994f},
                1e-4f);
    }
}
