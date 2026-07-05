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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import org.junit.jupiter.api.Test;

class ChatHistoryModeTest {

    @Test
    void defaultIsLlama() {
        assertEquals(ChatHistoryMode.LLAMA, ChatHistoryMode.parseName("llama"));
        assertEquals(ChatHistoryMode.LLAMA, ChatHistoryMode.parseName("llama.cpp"));
    }

    @Test
    void legacyAliases() {
        assertEquals(ChatHistoryMode.LEGACY, ChatHistoryMode.parseName("legacy"));
        assertEquals(ChatHistoryMode.LEGACY, ChatHistoryMode.parseName("tinygrad"));
        assertEquals(ChatHistoryMode.LEGACY, ChatHistoryMode.parseName("tensor4j"));
    }
}
