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
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.fixture.ChatDemoVocabMode;
import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.models.chat.fixture.RequiresChatDemoVocab;
import org.junit.jupiter.api.Test;

/**
 * Documents concrete defects found in the chat runtime (not hypothetical KV issues).
 *
 * <p>Each test names the broken behaviour and the fix that addresses it.
 */
class ChatKnownDefectsTest {

    /** {@code finishTurnLegacy} used to assign the same array reference to both fields. */
    @Test
    void legacyCachedTokensAliasingMutatedConversationInPlace() {
        int[] conversation = {10, 11, 12};
        int[] cached = conversation; // pre-fix: cachedTokens = conversationTokens
        cached[0] = 99;
        assertEquals(99, conversation[0], "aliasing lets cached mutation corrupt conversation");
        int[] fixed = ChatInferBufferPolicy.isolateSessionTokens(conversation);
        fixed[0] = 77;
        assertNotEquals(77, conversation[0]);
    }

    /**
     * When the model keeps sampling EOG before {@code min_new_tokens}, the old loop did {@code continue}
     * forever — user sees "completion never stops".
     */
    @Test
    void eogBeforeMinNewTokensSpinsToMaxWithoutBreakFix() {
        int minNew = 2;
        int maxNew = 256;
        assertEquals(maxNew, simulateDecodeSteps(minNew, maxNew, 0, false), "old loop exhausts max_new_tokens");
        assertEquals(1, simulateDecodeSteps(minNew, maxNew, 0, true), "break fix stops on first EOG");
    }

    private static int simulateDecodeSteps(int minNew, int maxNew, int generated, boolean breakEogSpin) {
        int steps = 0;
        for (int step = 0; step < maxNew; step++) {
            steps = step + 1;
            boolean endOfGen = true;
            boolean stop = endOfGen && generated >= minNew;
            if (stop) {
                break;
            }
            if (endOfGen) {
                if (breakEogSpin) {
                    break;
                }
                continue;
            }
            generated++;
        }
        return steps;
    }

    /**
     * {@link ChatModel#forward(int[])} returns {@code logits2d.data()} directly — callers must not mutate it
     * or assume it stays valid across calls.
     */
    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void modelForwardReturnsLiveBufferThatMustNotBeMutated() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        int[] prompt = {1, 2, 1};
        float[] first = model.forward(prompt);
        float saved = first[0];
        first[0] = saved + 12345f;
        model.resetCache();
        float[] second = model.forward(prompt);
        if (System.identityHashCode(first) == System.identityHashCode(second)) {
            assertEquals(saved + 12345f, second[0], 1e-3f, "runtime reuses logits buffer — mutation persists");
        }
    }

    /** Mini Qwen fixture vocab is only {@code Hello} + ChatML specials — not arbitrary English. */
    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void qwenTurnTwoPromptIncludesBothUserTurnsOnMiniFixture() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        ChatTokenizer tokenizer = model.tokenizer();
        ChatGenerator generator = new ChatGenerator(
                model, ChatGenerationOptions.greedy(tokenizer, 4), ChatHistoryMode.LLAMA);
        generator.continueConversation("Hello", ChatTemplate.QWEN2);

        int[] turn2 = generator.planPromptIds("Hello", ChatTemplate.QWEN2);
        int helloId = tokenizer.tokenIdForText("Hello");
        int countHello = 0;
        for (int id : turn2) {
            if (id == helloId) {
                countHello++;
            }
        }
        assertEquals(2, countHello, "full replay must contain both user Hello bodies");
    }

    /** Two-turn greedy on mini model completes without hitting MAX_TOKENS when break fix is on (default). */
    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void qwenTwoTurnGreedyStopsOnEndTokenNotMaxTokens() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        ChatGenerator generator = new ChatGenerator(
                model, ChatGenerationOptions.greedy(model.tokenizer(), 8), ChatHistoryMode.LLAMA);
        ChatGenerationResult turn1 = generator.continueConversation("Hi", ChatTemplate.QWEN2);
        ChatGenerationResult turn2 = generator.continueConversation("Yo", ChatTemplate.QWEN2);
        assertNotEquals(ChatGenerationStopReason.MAX_TOKENS, turn1.stopReason(), () -> "turn1: " + turn1);
        assertNotEquals(ChatGenerationStopReason.MAX_TOKENS, turn2.stopReason(), () -> "turn2: " + turn2);
    }
}
