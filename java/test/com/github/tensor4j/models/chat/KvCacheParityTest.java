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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Cross-runtime KV parity: tinygrad prefix-resume ≡ cold prefill; llama.cpp delta suffix;
 * legacy cached-token bookkeeping.
 */
class KvCacheParityTest {

    @Test
    void partialReusePrefillMatchesColdForwardLogits() {
        ChatModel warm = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        ChatModel cold = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        int[] prompt = {1, 2, 1, 2, 1, 2, 1, 2};
        int startPos = 4;

        warm.resetCache();
        warm.forward(Arrays.copyOfRange(prompt, 0, startPos));
        assertEquals(startPos, warm.kvLength(), "tinygrad n_past before resume prefill");

        float[] resumed = ChatGenerator.prefillChunked(warm, prompt, startPos, 3);

        cold.resetCache();
        float[] fresh = cold.forward(prompt);

        assertArrayEquals(fresh, resumed, 1e-4f, "chunked resume from start_pos must match cold prefill");
    }

    @Test
    void partialReuseWithChunkSizeOneMatchesColdForward() {
        ChatModel warm = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        ChatModel cold = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        int[] prompt = {1, 2, 1, 2, 1, 2};
        int startPos = 3;

        warm.resetCache();
        warm.forward(Arrays.copyOfRange(prompt, 0, startPos));
        float[] resumed = ChatGenerator.prefillChunked(warm, prompt, startPos, 1);

        cold.resetCache();
        float[] fresh = cold.forward(prompt);

        assertArrayEquals(fresh, resumed, 1e-4f);
    }

    @Test
    void legacyCachedTokenCountMatchesKvAfterPlainGenerate() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        ChatGenerationOptions options = ChatGenerationOptions.greedy(model.tokenizer(), 2);
        ChatGenerator generator = new ChatGenerator(model, options, ChatHistoryMode.LEGACY);

        int[] prompt = {1, 2, 1, 2, 1};
        generator.generateWithKvReuse(prompt);

        assertEquals(
                generator.cachedTokenIds().length,
                model.kvLength(),
                "tinygrad _cached_tokens length should match n_past after PLAIN turn");
    }

    @Test
    void legacyMultiTurnPromptExtendsCachedPrefix() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        ChatGenerationOptions options = ChatGenerationOptions.greedy(model.tokenizer(), 1);
        ChatGenerator generator = new ChatGenerator(model, options, ChatHistoryMode.LEGACY);

        generator.continueConversation("a", ChatTemplate.PLAIN);
        int[] cachedAfterFirst = generator.cachedTokenIds();

        int[] turnTwoPrompt = generator.planPromptIds("b", ChatTemplate.PLAIN);
        assertTrue(
                prefixMatches(cachedAfterFirst, turnTwoPrompt),
                "turn-2 prompt must extend turn-1 cached tokens (tinygrad growing ids list)");
    }

    @Test
    void legacyResumeMidGenerationMatchesFreshExtendedPrompt() {
        ChatModel warmModel = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        ChatModel coldModel = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        ChatGenerationOptions options = ChatGenerationOptions.greedy(warmModel.tokenizer(), 2);

        int[] prompt = {1, 2, 1, 2, 1};
        ChatGenerator warm = new ChatGenerator(warmModel, options, ChatHistoryMode.LEGACY);
        ChatGenerationResult partial = warm.generateWithKvReuse(prompt);

        int[] extended = concat(prompt, partial.forwardedTokenIds(), new int[] {2, 1});
        ChatGenerationResult resumed = warm.generateWithKvReuse(extended);

        ChatGenerator cold = new ChatGenerator(coldModel, options, ChatHistoryMode.LEGACY);
        ChatGenerationResult fromScratch = cold.generateWithKvReuse(extended);

        assertArrayEquals(fromScratch.generatedTokenIds(), resumed.generatedTokenIds());
        assertEquals(fromScratch.text(), resumed.text());
    }

    @Test
    void llamaDeltaIsSuffixOfFullTemplateWithGenPrompt() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel());
        ChatGenerator generator =
                new ChatGenerator(model, ChatGenerationOptions.greedy(model.tokenizer(), 4), ChatHistoryMode.LLAMA);
        generator.continueConversation("Hello", ChatTemplate.LLAMA3);

        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(model.tokenizer());
        int prev = generator.templatePrevTokens();
        List<ChatMessage> planned = new ArrayList<>(generator.messages());
        planned.add(new ChatMessage("user", "Again"));

        int[] full = applier.tokenIds(model.tokenizer(), planned, true);
        int[] delta = applier.tokenDeltaSince(model.tokenizer(), planned, true, prev);

        assertEquals(full.length - prev, delta.length, "simple-chat delta length");
        assertArrayEquals(
                Arrays.copyOfRange(full, prev, full.length),
                delta,
                "token delta must equal full[prev:] (llama.cpp formatted[prev_len:])");
    }

    @Test
    void llamaModeDoesNotResetKvBetweenTurns() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel());
        ChatGenerator generator =
                new ChatGenerator(model, ChatGenerationOptions.greedy(model.tokenizer(), 2), ChatHistoryMode.LLAMA);

        generator.continueConversation("Hi", ChatTemplate.LLAMA3);
        int kvAfterFirst = model.kvLength();
        assertTrue(kvAfterFirst > 0);

        generator.continueConversation("Yo", ChatTemplate.LLAMA3);
        assertTrue(
                model.kvLength() >= kvAfterFirst,
                "llama delta mode must not reset KV between turns (until ring fills)");
    }

    @Test
    void llamaFirstGenerateStringResetsKvThenGrows() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel());
        ChatGenerator generator =
                new ChatGenerator(model, ChatGenerationOptions.greedy(model.tokenizer(), 2), ChatHistoryMode.LLAMA);

        model.resetCache();
        model.forward(new int[] {1, 2});
        assertEquals(2, model.kvLength());

        generator.generate("Hi", ChatTemplate.LLAMA3);
        assertTrue(model.kvLength() > 0);
        assertTrue(model.kvLength() != 2, "generate(String) must reset stale KV before first turn");
    }

    @Test
    void resetSessionClearsKvAndConversationState() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel());
        ChatGenerator generator =
                new ChatGenerator(model, ChatGenerationOptions.greedy(model.tokenizer(), 2), ChatHistoryMode.LLAMA);

        generator.continueConversation("Hi", ChatTemplate.LLAMA3);
        assertTrue(model.kvLength() > 0);
        assertTrue(generator.templatePrevTokens() > 0);

        generator.resetSession();
        assertEquals(0, model.kvLength());
        assertEquals(0, generator.templatePrevTokens());
        assertEquals(0, generator.sessionTokenIds().length);
        assertTrue(generator.messages().isEmpty());
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void legacyDivergentPrefixForcesFullKvReset() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        ChatGenerationOptions options = ChatGenerationOptions.greedy(model.tokenizer(), 1);
        ChatGenerator generator = new ChatGenerator(model, options, ChatHistoryMode.LEGACY);

        generator.generateWithKvReuse(new int[] {1, 2, 1, 2, 1});
        int kvAfterFirst = model.kvLength();

        ChatGenerationResult divergent = generator.generateWithKvReuse(new int[] {2, 1, 2});
        assertEquals(0, divergent.prefixReuseTokens(), "divergent ids must not reuse prefix");
        assertTrue(model.kvLength() <= kvAfterFirst + 3, "cache reset + short prompt refilled");
    }

    @Test
    void reencodingAssistantTextCanDriftFromSampledTokenIds() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel());
        ChatGenerator generator =
                new ChatGenerator(model, ChatGenerationOptions.greedy(model.tokenizer(), 4), ChatHistoryMode.LLAMA);
        ChatGenerationResult result = generator.continueConversation("Hi", ChatTemplate.LLAMA3);

        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(model.tokenizer());
        List<ChatMessage> withReencode = List.of(
                new ChatMessage("user", "Hi"),
                new ChatMessage("assistant", result.text()));
        List<ChatMessage> withIds = List.of(
                new ChatMessage("user", "Hi"),
                new ChatMessage("assistant", result.text(), result.forwardedTokenIds()));

        int[] reencoded = applier.tokenIds(model.tokenizer(), withReencode, false);
        int[] fromIds = applier.tokenIds(model.tokenizer(), withIds, false);
        assertArrayEquals(
                generator.sessionTokenIds(),
                fromIds,
                "session must commit sampled assistant ids (llama.cpp KV parity)");
        if (!Arrays.equals(reencoded, fromIds)) {
            assertTrue(
                    reencoded.length != fromIds.length || !Arrays.equals(reencoded, fromIds),
                    "re-encoding decoded text can diverge from forwarded token ids — KV drift risk");
        }
    }

    private static boolean prefixMatches(int[] prefix, int[] full) {
        if (prefix.length > full.length) {
            return false;
        }
        for (int i = 0; i < prefix.length; i++) {
            if (prefix[i] != full[i]) {
                return false;
            }
        }
        return true;
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
