/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.chat.demo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.ChatGenerationOptions;
import com.github.tensor4j.models.chat.ChatGenerationResult;
import com.github.tensor4j.models.chat.ChatGenerator;
import com.github.tensor4j.models.chat.ChatModel;
import com.github.tensor4j.models.chat.ChatTemplate;
import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder.ChatDemo;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.api.Test;

/** Four-turn user/bot session with KV prefix reuse (level-12 demo). */
class FourTurnChatDemoIT {

    @Test
    @EnabledIf("com.github.tensor4j.models.chat.fixture.ChatDemoVocab#fullFixturePresent")
    void fourTurnHelloConversationReusesKv() throws Exception {
        ChatDemoReporter.banner("four-turn chat (level 12)");
        ChatModel model = ChatSession.loadOpenModel();
        ChatGenerationOptions options = ChatSession.optionsFor(model);
        ChatDemoReporter.modelInfo(model, "MiniChatGgufBuilder.buildOpenChatDemoModel()");

        assertTrue(model.config().nEmbd() == ChatDemo.N_EMBD);
        assertTrue(model.config().nVocab() == ChatDemo.FULL_VOCAB);

        ChatGenerator generator = new ChatGenerator(model, options);
        int lastReuse = 0;
        for (int turn = 1; turn <= ChatDemo.TURN_COUNT; turn++) {
            String user = "Hello";
            ChatDemoReporter.tokenize(model.tokenizer(), user);
            ChatGenerationResult result = generator.continueConversation(user, ChatTemplate.PLAIN);
            ChatDemoReporter.generation(
                    user, result.text(), result.tokenCount(), result.mode(), result.prefixReuseTokens());
            ChatSession.assertRealCompletion(result.text());
            if (turn == 1) {
                assertEquals(0, result.prefixReuseTokens());
            } else {
                assertTrue(result.prefixReuseTokens() > lastReuse, "turn " + turn + " should grow KV reuse");
            }
            lastReuse = result.prefixReuseTokens();
        }
        ChatDemoReporter.summary(ChatDemo.TURN_COUNT, ChatDemo.TURN_COUNT);
    }
}
