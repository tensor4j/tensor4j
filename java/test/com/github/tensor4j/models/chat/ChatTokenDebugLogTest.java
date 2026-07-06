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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import org.junit.jupiter.api.Test;

class ChatTokenDebugLogTest {

    @Test
    void enabledByDefaultWhenEnvUnset() {
        assertTrue(ChatTokenDebugLog.parseEnabled(null));
        assertTrue(ChatTokenDebugLog.parseEnabled(""));
        assertTrue(ChatTokenDebugLog.parseEnabled("true"));
        assertFalse(ChatTokenDebugLog.parseEnabled("false"));
    }

    @Test
    void formatIdsWrapsLongRuns() {
        int[] ids = new int[40];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = 151644 + (i % 5);
        }
        String formatted = ChatTokenDebugLog.formatIds(ids);
        assertTrue(formatted.contains("\n "), () -> "expected wrapped ids: " + formatted);
        assertTrue(formatted.startsWith("["));
        assertTrue(formatted.endsWith("]"));
    }

    @Test
    void promptDecodeLabelReflectsTurnAndMode() {
        assertEquals("prompt_decode_prefill", ChatTokenDebugLog.promptDecodeLabel(0, false));
        assertEquals("prompt_decode_turn1", ChatTokenDebugLog.promptDecodeLabel(0, true));
        assertEquals("prompt_decode_delta", ChatTokenDebugLog.promptDecodeLabel(42, true));
    }

    @Test
    void formatTokenMapIncludesImEndPiece() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel().header());
        int[] endTurn = ChatTemplate.QWEN2.encodeEndTurn(tokenizer);
        String map = ChatTokenDebugLog.formatTokenMap(tokenizer, endTurn);
        assertTrue(map.contains(Integer.toString(tokenizer.endTurnId())));
        assertTrue(map.contains(tokenizer.tokenText(tokenizer.endTurnId())));
    }
}
