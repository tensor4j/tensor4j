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
import com.github.tensor4j.models.chat.fixture.RequiresChatDemoVocab;
import com.github.tensor4j.models.chat.fixture.ChatDemoVocabMode;
import com.github.tensor4j.models.chat.reference.TinygradGenerateReference;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

/**
 * KV position / prefix indexing — aligned with llama.cpp {@code n_past} and tinygrad
 * {@code get_start_pos()}.
 */
class KvCacheIndexingTest {

    @Test
    void getStartPosExcludesLastPromptToken() {
        int[] prompt = {1, 2, 3, 4};
        int[] cached = {1, 2, 3, 4, 9};
        assertEquals(3, ChatGenerator.sharedPrefixLength(prompt, cached));
        assertEquals(3, TinygradGenerateReference.getStartPos(prompt, cached));
    }

    @Test
    void getStartPosIsZeroForEmptyCacheOrSingleTokenPrompt() {
        assertEquals(0, ChatGenerator.sharedPrefixLength(new int[] {5}, new int[0]));
        assertEquals(0, ChatGenerator.sharedPrefixLength(new int[] {5}, new int[] {5, 6}));
    }

    @Test
    void getStartPosStopsAtFirstMismatch() {
        int[] prompt = {1, 2, 9, 4};
        int[] cached = {1, 2, 3, 4};
        assertEquals(2, ChatGenerator.sharedPrefixLength(prompt, cached));
    }

    @Test
    void getStartPosFullPrefixExceptLastMatchesCachedLength() {
        int[] prompt = {1, 2, 3, 4, 5};
        int[] cached = {1, 2, 3, 4};
        assertEquals(4, ChatGenerator.sharedPrefixLength(prompt, cached));
    }

    @Test
    void prefillSliceLengthsWithMidStartPosCoverTailOnly() {
        assertArrayEquals(new int[] {2, 2, 2}, ChatGenerator.prefillSliceLengths(9, 3, 2));
        assertArrayEquals(new int[] {5}, ChatGenerator.prefillSliceLengths(8, 3, 5));
    }

    @Test
    void prefillWithStartPosOnlyAppendsTailToKv() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        int[] prompt = {1, 2, 1, 2, 1, 2};
        model.resetCache();
        model.forward(new int[] {1, 2, 1});
        assertEquals(3, model.kvLength());

        ChatGenerator.prefillChunked(model, prompt, 3, 2);
        assertEquals(prompt.length, model.kvLength(), "only tokens[startPos:] should extend n_past");
    }

    @Test
    void llamaModeFinishTurnInjectsEotWhenGenerationDidNotStopOnEog() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel());
        ChatGenerator generator =
                new ChatGenerator(model, ChatGenerationOptions.greedy(model.tokenizer(), 2), ChatHistoryMode.LLAMA);
        ChatGenerationResult result = generator.continueConversation("Hi", ChatTemplate.LLAMA3);
        if (result.stopReason() == ChatGenerationStopReason.EOT
                || result.stopReason() == ChatGenerationStopReason.EOS) {
            return;
        }
        int closed = generator.templatePrevTokens();
        if (closed <= model.config().nCtx()) {
            assertEquals(closed, model.kvLength(), "EOT inject should keep KV aligned with closed template");
        }
    }

    @Test
    void llamaModeCachedTokensEqualClosedTemplateAfterTurn() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel());
        ChatGenerator generator =
                new ChatGenerator(model, ChatGenerationOptions.greedy(model.tokenizer(), 4), ChatHistoryMode.LLAMA);
        generator.continueConversation("Hi", ChatTemplate.LLAMA3);
        assertArrayEquals(generator.sessionTokenIds(), generator.cachedTokenIds());
    }

    @Test
    void forwardAdvancesKvByBatchSize() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        model.resetCache();
        assertEquals(0, model.kvLength(), "llama.cpp n_past=0 on empty cache");

        model.forward(new int[] {1, 2});
        assertEquals(2, model.kvLength());

        model.forward(new int[] {1});
        assertEquals(3, model.kvLength(), "single decode step appends at n_past=2");
    }

    @Test
    void tinygradGetStartPosMatchesSharedPrefixLength() {
        int[] prompt = {1, 2, 3, 4, 5};
        int[] cached = {1, 2, 9, 9};
        assertEquals(
                TinygradGenerateReference.getStartPos(prompt, cached),
                ChatGenerator.sharedPrefixLength(prompt, cached));
    }

    @Test
    void chunkedPrefillAdvancesKvLikeLlamaCppBatchDecode() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        int[] prompt = {1, 2, 1, 2, 1, 2, 1, 2};
        int[] slices = ChatGenerator.prefillSliceLengths(prompt.length, 0, 3);
        model.resetCache();
        int pos = 0;
        for (int slice : slices) {
            model.forward(Arrays.copyOfRange(prompt, pos, pos + slice));
            pos += slice;
            assertEquals(pos, model.kvLength(), "each chunk must append at contiguous n_past");
        }
        assertArrayEquals(new int[] {3, 3, 2}, slices);
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void legacyPrefixReuseSkipsPrefillAtStartPos() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        ChatGenerationOptions options = ChatGenerationOptions.greedy(model.tokenizer(), 2);
        ChatGenerator generator = new ChatGenerator(model, options, ChatHistoryMode.LEGACY);

        int[] prompt = {1, 2, 1, 2, 1};
        ChatGenerationResult first = generator.generateWithKvReuse(prompt);
        assertEquals(0, first.prefixReuseTokens());
        int kvAfterFirst = model.kvLength();
        assertTrue(kvAfterFirst > prompt.length, "generation should extend KV beyond prompt");

        int[] extended = concat(prompt, first.forwardedTokenIds(), new int[] {2, 1});
        int reuse = ChatGenerator.sharedPrefixLength(extended, generator.cachedTokenIds());
        assertEquals(kvAfterFirst, reuse, "tinygrad start_pos should match cached prompt prefix");

        ChatGenerationResult second = generator.generateWithKvReuse(extended);
        assertEquals(reuse, second.prefixReuseTokens());
        assertTrue(model.kvLength() > kvAfterFirst);
    }

    @Test
    void llamaModeKvMatchesClosedTemplateWhenWithinContextWindow() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel());
        ChatGenerator generator =
                new ChatGenerator(model, ChatGenerationOptions.greedy(model.tokenizer(), 4), ChatHistoryMode.LLAMA);

        generator.continueConversation("Hi", ChatTemplate.LLAMA3);
        int closedLen = generator.templatePrevTokens();
        int kv = model.kvLength();
        if (closedLen <= model.config().nCtx()) {
            assertEquals(closedLen, kv, "n_past must match closed template when all tokens fit in n_ctx");
        } else {
            assertEquals(model.config().nCtx(), kv, "ring KV caps at n_ctx when template exceeds window");
        }

        int closedAfterFirst = closedLen;
        generator.continueConversation("Yo", ChatTemplate.LLAMA3);
        assertTrue(
                generator.templatePrevTokens() > closedAfterFirst,
                "multi-turn delta mode must grow closed template");
    }

    @Test
    void llamaModeStoresAssistantGenerationTokenIds() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel());
        ChatGenerator generator =
                new ChatGenerator(model, ChatGenerationOptions.greedy(model.tokenizer(), 8), ChatHistoryMode.LLAMA);

        ChatGenerationResult result = generator.continueConversation("Hello", ChatTemplate.LLAMA3);
        ChatMessage assistant = generator.messages().get(1);
        assertEquals("assistant", assistant.role());
        assertArrayEquals(
                result.forwardedTokenIds(),
                assistant.generatedTokenIds(),
                "assistant turn must keep sampled ids, not re-encode decoded text");
    }

    @Test
    void llamaEotStopReasonWhenEosAndEotShareId() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel().header());
        int shared = tokenizer.eosId();
        assertEquals(
                ChatGenerationStopReason.EOT,
                ChatGenerationStopReason.forEndToken(tokenizer, shared),
                "Llama 3 chat stops on <|eot_id|> even when eos_id matches");
    }

    private static int[] concat(int[]... parts) {
        int len = 0;
        for (int[] part : parts) {
            len += part.length;
        }
        int[] out = new int[len];
        int pos = 0;
        for (int[] part : parts) {
            System.arraycopy(part, 0, out, pos, part.length);
            pos += part.length;
        }
        return out;
    }
}
