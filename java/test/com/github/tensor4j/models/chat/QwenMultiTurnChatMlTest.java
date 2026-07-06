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
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Multi-turn Qwen ChatML: turn-2 delta must extend a system-first, properly closed session. */
class QwenMultiTurnChatMlTest {

    private static final String IM_START = "<|im_start|>";

    @Test
    void turnTwoDeltaPrefillIsValidChatMlAfterSystemFirstSession() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        ChatTokenizer tokenizer = model.tokenizer();
        ChatGenerator generator = new ChatGenerator(
                model, ChatGenerationOptions.greedy(tokenizer, 16), ChatHistoryMode.LLAMA);

        generator.continueConversation("write a java hello world program", ChatTemplate.QWEN2);
        if (generator.sessionTokenIds().length > model.config().nCtx()) {
            return;
        }
        assertSessionMatchesKvAndTemplate(generator);

        int[] sessionBefore = generator.sessionTokenIds();
        int[] promptDelta = generator.planPromptIds("explain spring vs java ee", ChatTemplate.QWEN2);
        assertTrue(promptDelta.length > 0, "turn 2 must prefill new ChatML tokens");

        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);
        List<ChatMessage> planned = new ArrayList<>(generator.messages());
        planned.add(new ChatMessage("user", "explain spring vs java ee"));
        int[] full = applier.tokenIds(tokenizer, planned, true);
        assertArrayEquals(full, concat(sessionBefore, promptDelta));

        String deltaDecode = tokenizer.decode(promptDelta);
        assertTrue(
                deltaDecode.startsWith(IM_START + "user\n"),
                () -> "turn 2 delta must open a new user ChatML turn, got: " + deltaDecode);

        String sessionDecode = tokenizer.decode(sessionBefore);
        assertTrue(
                sessionDecode.startsWith(IM_START + "system\n"),
                () -> "session must remain system-first ChatML, got: " + sessionDecode);
        assertChatMlTurnClosed(sessionBefore, tokenizer);

        generator.continueConversation("explain spring vs java ee", ChatTemplate.QWEN2);
        assertSessionMatchesKvAndTemplate(generator);

        String fullSessionDecode = tokenizer.decode(generator.sessionTokenIds());
        assertTrue(fullSessionDecode.contains(IM_START + "user\nexplain spring vs java ee"));
    }

    @Test
    void turnTwoApplierDeltaMatchesClosedAssistantTurnWithSampledIds() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel().header());
        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);
        int imEnd = tokenizer.endTurnId();
        int newline = newlineId(tokenizer);

        int[] assistantBody = new int[] {1, 1, 1};
        int[] assistantForwarded = concat(assistantBody, new int[] {imEnd});
        List<ChatMessage> afterTurnOne = List.of(
                new ChatMessage("system", ChatTemplate.QWEN2_DEFAULT_SYSTEM),
                new ChatMessage("user", "hello"),
                new ChatMessage("assistant", "aaa", assistantForwarded));

        int prev = applier.tokenCountAfterAssistantTurn(tokenizer, afterTurnOne);
        List<ChatMessage> turnTwo = List.of(
                afterTurnOne.get(0),
                afterTurnOne.get(1),
                afterTurnOne.get(2),
                new ChatMessage("user", "second"));
        int[] closed = applier.tokenIds(tokenizer, afterTurnOne, false);
        int[] full = applier.tokenIds(tokenizer, turnTwo, true);
        int[] delta = applier.tokenDeltaSince(tokenizer, turnTwo, true, prev);

        assertEquals(imEnd, closed[closed.length - 2], "turn 1 must close with im_end before newline");
        assertEquals(newline, closed[closed.length - 1], "turn 1 must end with newline after im_end");
        assertArrayEquals(java.util.Arrays.copyOfRange(full, prev, full.length), delta);
        assertEquals(tokenizer.tokenIdForText(IM_START), delta[0]);
        assertEquals(tokenizer.tokenIdForText("user"), delta[1]);
        assertEquals(newline, delta[2], "user role header must end with newline token");
    }

    private static void assertSessionMatchesKvAndTemplate(ChatGenerator generator) {
        assertEquals(generator.sessionTokenIds().length, generator.templatePrevTokens(), "session must match template cursor");
        assertEquals(generator.sessionTokenIds().length, generator.kvLength(), "KV must match session on full-context models");
        assertArrayEquals(generator.sessionTokenIds(), generator.cachedTokenIds(), "cached must match session");
    }

    private static void assertChatMlTurnClosed(int[] session, ChatTokenizer tokenizer) {
        int imEnd = tokenizer.endTurnId();
        int newline = newlineId(tokenizer);
        int imEndIndex = lastIndexOf(session, imEnd);
        assertTrue(imEndIndex >= 0, "closed session must contain im_end");
        assertEquals(newline, session[imEndIndex + 1], "each closed turn must end im_end then newline");
    }

    private static int newlineId(ChatTokenizer tokenizer) {
        int[] encoded = tokenizer.encode("\n");
        return encoded.length > 0 ? encoded[0] : tokenizer.tokenIdForText("\n");
    }

    private static int lastIndexOf(int[] array, int value) {
        for (int i = array.length - 1; i >= 0; i--) {
            if (array[i] == value) {
                return i;
            }
        }
        return -1;
    }

    private static int[] concat(int[] left, int[] right) {
        int[] out = new int[left.length + right.length];
        System.arraycopy(left, 0, out, 0, left.length);
        System.arraycopy(right, 0, out, left.length, right.length);
        return out;
    }
}
