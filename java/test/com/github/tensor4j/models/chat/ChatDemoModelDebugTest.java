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

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.models.chat.fixture.RequiresChatDemoVocab;
import com.github.tensor4j.models.chat.fixture.ChatDemoVocabMode;
import org.junit.jupiter.api.Test;

class ChatDemoModelDebugTest {

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.FULL)
    void helloProducesRealCompletionUnderQualitySampling() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildOpenChatDemoModel());
        ChatGenerator generator = new ChatGenerator(model, ChatGenerationOptions.quality(model.tokenizer()));
        ChatGenerationResult result = generator.generate("Hello", ChatTemplate.PLAIN);
        assertTrue(result.text().length() > 1, result::text);
    }
}
