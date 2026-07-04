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

import com.github.tensor4j.models.chat.ChatGenerationOptions;
import com.github.tensor4j.models.chat.ChatGenerator;
import com.github.tensor4j.models.chat.ChatModel;
import com.github.tensor4j.models.chat.ChatTemplate;
import com.github.tensor4j.models.chat.fixture.ChatDemoVocabMode;
import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import java.util.Arrays;

/** Prints greedy decode token ids for intermediate golden check-in. */
public final class GenerateGoldenProbe {

    private GenerateGoldenProbe() {
    }

    public static void main(String[] args) {
        ChatModel chain = ChatModel.fromGguf(MiniChatGgufBuilder.buildChatDemoModel());
        ChatGenerator chainGen = new ChatGenerator(chain, ChatGenerationOptions.greedy(chain.tokenizer(), 5));
        System.out.println("chain_40="
                + Arrays.toString(chainGen.generate(new int[] {40}).generatedTokenIds()));

        ChatModel identity = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        ChatGenerator idGen = new ChatGenerator(identity, ChatGenerationOptions.greedy(identity.tokenizer(), 3));
        var plain = idGen.generate("ab", ChatTemplate.PLAIN);
        System.out.println("identity_ab_greedy_ids=" + Arrays.toString(plain.generatedTokenIds()));
        System.out.println("identity_ab_greedy_text=" + plain.text());
    }
}
