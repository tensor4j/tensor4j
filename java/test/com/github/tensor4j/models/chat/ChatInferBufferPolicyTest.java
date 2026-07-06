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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.models.chat.fixture.RequiresChatDemoVocab;
import com.github.tensor4j.models.chat.fixture.ChatDemoVocabMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Isolation tests for prompt / logits / session buffer leaks between inference calls. */
class ChatInferBufferPolicyTest {

    @Test
    void bufferPolicyDefaultsFavourDefensiveCopies() {
        assertTrue(ChatInferBufferPolicy.copyPromptTokens());
        assertTrue(ChatInferBufferPolicy.cloneForwardLogits());
        assertTrue(ChatInferBufferPolicy.defensiveSessionCopy());
        assertFalse(ChatInferBufferPolicy.logBufferIdentity());
        assertFalse(ChatInferBufferPolicy.logPromptTextBeforeTokenize());
    }

    @Test
    void isolatePromptTokensDecouplesFromSourceMutation() {
        int[] source = {1, 2, 3};
        int[] isolated = ChatInferBufferPolicy.isolatePromptTokens(source);
        source[0] = 99;
        if (ChatInferBufferPolicy.copyPromptTokens()) {
            assertEquals(1, isolated[0]);
        } else {
            assertEquals(99, isolated[0]);
        }
    }

    @Test
    void isolateForwardLogitsDecouplesFromSourceMutation() {
        float[] source = {1f, 2f, 3f};
        float[] isolated = ChatInferBufferPolicy.isolateForwardLogits(source);
        source[0] = 99f;
        if (ChatInferBufferPolicy.cloneForwardLogits()) {
            assertEquals(1f, isolated[0], 0f);
        } else {
            assertEquals(99f, isolated[0], 0f);
        }
    }

    @Test
    void isolateSessionTokensDecouplesFromSourceMutation() {
        int[] source = {4, 5, 6};
        int[] isolated = ChatInferBufferPolicy.isolateSessionTokens(source);
        source[1] = 99;
        if (ChatInferBufferPolicy.defensiveSessionCopy()) {
            assertEquals(5, isolated[1]);
        } else {
            assertEquals(99, isolated[1]);
        }
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void forwardTrackedIsolatesStoredLogitsFromLaterMutation() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        ChatGenerator generator =
                new ChatGenerator(model, ChatGenerationOptions.greedy(model.tokenizer(), 2), ChatHistoryMode.LLAMA);
        int[] prompt = {1, 2, 1};
        float[] stored = generator.forwardTrackedForTests(prompt);
        float before = stored[0];
        stored[0] = before - 999f;
        model.resetCache();
        float[] fresh = generator.forwardTrackedForTests(prompt);
        assertNotEquals(before - 999f, fresh[0], 1e-3f);
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void mutatingPrefillPromptIdsDoesNotChangeCommittedPlan() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        ChatGenerator generator =
                new ChatGenerator(model, ChatGenerationOptions.greedy(model.tokenizer(), 2), ChatHistoryMode.LLAMA);
        generator.continueConversation("one", ChatTemplate.QWEN2);

        int[] planned = generator.planPromptIds("two", ChatTemplate.QWEN2);
        int[] snapshot = planned.clone();
        planned[0] = -1;
        assertArrayEquals(snapshot, generator.planPromptIds("two", ChatTemplate.QWEN2));
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void turnTwoPromptMatchesFreshApplierBuildNotStaleBuffer() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        ChatTokenizer tokenizer = model.tokenizer();
        ChatGenerator generator =
                new ChatGenerator(model, ChatGenerationOptions.greedy(tokenizer, 2), ChatHistoryMode.LLAMA);
        generator.continueConversation("one", ChatTemplate.QWEN2);

        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);
        List<ChatMessage> freshMessages = new ArrayList<>(generator.messages());
        freshMessages.add(new ChatMessage("user", "two"));
        int[] rebuilt = applier.tokenIds(tokenizer, freshMessages, true);
        int[] planned = generator.planPromptIds("two", ChatTemplate.QWEN2);
        if (ChatGenerator.kvCacheEnabled()) {
            int prev = generator.templatePrevTokens();
            assertArrayEquals(Arrays.copyOfRange(rebuilt, prev, rebuilt.length), planned);
        } else {
            assertArrayEquals(rebuilt, planned);
        }
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void identicalPromptTwiceYieldsIdenticalGeneration() {
        ChatModel warm = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        ChatModel fresh = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        ChatGenerationOptions options = ChatGenerationOptions.greedy(warm.tokenizer(), 4);

        ChatGenerator firstRun = new ChatGenerator(warm, options, ChatHistoryMode.LLAMA);
        ChatGenerator secondRun = new ChatGenerator(fresh, options, ChatHistoryMode.LLAMA);

        ChatGenerationResult a = firstRun.continueConversation("Hi", ChatTemplate.QWEN2);
        ChatGenerationResult b = secondRun.continueConversation("Hi", ChatTemplate.QWEN2);

        assertArrayEquals(a.forwardedTokenIds(), b.forwardedTokenIds());
        assertEquals(a.text(), b.text());
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void resetSessionClearsTurnStateForRepeatPrompt() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        ChatGenerationOptions options = ChatGenerationOptions.greedy(model.tokenizer(), 4);
        ChatGenerator generator = new ChatGenerator(model, options, ChatHistoryMode.LLAMA);

        ChatGenerationResult turnOne = generator.continueConversation("Hi", ChatTemplate.QWEN2);
        generator.resetSession();
        ChatGenerationResult turnOneAgain = generator.continueConversation("Hi", ChatTemplate.QWEN2);

        assertArrayEquals(turnOne.forwardedTokenIds(), turnOneAgain.forwardedTokenIds());
        assertEquals(turnOne.text(), turnOneAgain.text());
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void legacyCachedTokensNotAliasedToConversationTokens() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        ChatGenerator generator =
                new ChatGenerator(model, ChatGenerationOptions.greedy(model.tokenizer(), 2), ChatHistoryMode.LEGACY);
        generator.continueConversation("Hi", ChatTemplate.QWEN2);

        int[] cached = generator.cachedTokensInternal();
        int[] conversation = generator.conversationTokensInternal();
        if (ChatInferBufferPolicy.defensiveSessionCopy()) {
            assertNotSame(cached, conversation);
        }
        cached[0] = -1;
        assertNotEquals(-1, generator.conversationTokensInternal()[0]);
    }
}
