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

import com.github.tensor4j.models.chat.ChatModel;
import com.github.tensor4j.models.chat.ChatSampler;
import com.github.tensor4j.models.chat.ChatTemplate;
import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.runtime.gguf.GgufTensorSource;
import java.util.Arrays;

/** Prints forward logits for intermediate golden check-in. */
public final class ForwardGoldenProbe {

    private ForwardGoldenProbe() {
    }

    public static void main(String[] args) {
        probe("llama3_bpe_hello", MiniChatGgufBuilder.buildLlama3BpeModel(), new int[] {1});
        probe("identity_a", MiniChatGgufBuilder.buildIdentityModel(), new int[] {1});
        probe("identity_ab", MiniChatGgufBuilder.buildIdentityModel(), new int[] {1, 2});

        GgufTensorSource templateFixture = MiniChatGgufBuilder.buildLlama3TemplateModel();
        ChatModel templateModel = ChatModel.fromGguf(templateFixture);
        int[] userHello = ChatTemplate.LLAMA3.encodeUser(templateModel.tokenizer(), "Hello");
        System.out.println("llama3_template_ids=" + Arrays.toString(userHello));
        probe("llama3_template_user_hello", templateFixture, userHello);

        ChatModel identity = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        identity.resetCache();
        float[] inc = identity.forward(new int[] {1});
        inc = identity.forward(new int[] {2});
        System.out.println("identity_ab_incremental=" + Arrays.toString(inc));
    }

    private static void probe(String label, GgufTensorSource source, int[] tokens) {
        ChatModel model = ChatModel.fromGguf(source);
        float[] logits = model.forward(tokens);
        System.out.println(label + " logits=" + Arrays.toString(logits));
        System.out.println(label + " argmax=" + ChatSampler.argmax(logits));
    }
}
