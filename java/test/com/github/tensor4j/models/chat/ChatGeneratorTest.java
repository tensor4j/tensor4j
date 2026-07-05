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

import java.util.Arrays;
import com.github.tensor4j.models.chat.ChatSamplingRngMode;
import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.models.chat.fixture.RequiresChatDemoVocab;
import com.github.tensor4j.models.chat.fixture.ChatDemoVocabMode;
import org.junit.jupiter.api.Test;

class ChatGeneratorTest {

    private static ChatGenerationOptions greedyWithPrefillChunk(ChatTokenizer tokenizer, int maxNewTokens, int chunk) {
        return new ChatGenerationOptions(
                ChatGenerationMode.GREEDY,
                0f,
                1f,
                0,
                0,
                maxNewTokens,
                0L,
                tokenizer.bosId(),
                tokenizer.eosId(),
                tokenizer.eotId(),
                0f,
                0f,
                false,
                chunk,
                ChatSamplingRngMode.LEGACY);
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

    @Test
    void sharedPrefixLengthMatchesTinygradGetStartPos() {
        assertEquals(0, ChatGenerator.sharedPrefixLength(new int[] {1}, new int[] {1, 2}));
        assertEquals(2, ChatGenerator.sharedPrefixLength(new int[] {1, 2, 9}, new int[] {1, 2, 3}));
        assertEquals(0, ChatGenerator.sharedPrefixLength(new int[] {9, 2}, new int[] {1, 2}));
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void chatDemoModelHasLevel12Scale() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildChatDemoModel());
        assertEquals(MiniChatGgufBuilder.ChatDemo.N_EMBD, model.config().nEmbd());
        assertEquals(MiniChatGgufBuilder.ChatDemo.N_LAYER, model.config().nLayer());
        assertEquals(MiniChatGgufBuilder.ChatDemo.N_CTX, model.config().nCtx());
        assertTrue(model.config().nVocab() >= MiniChatGgufBuilder.ChatDemo.PRUNED_MIN_VOCAB);
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void chunkedPrefillMatchesSingleForward() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildChatDemoModel());
        int[] prompt = model.tokenizer().encode("Hello");
        model.resetCache();
        float[] single = model.forward(prompt);
        model.resetCache();
        float[] chunked = ChatGenerator.prefillChunked(model, prompt, 0, 1);
        assertArrayEquals(single, chunked, 1e-4f);
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void seededQualityGenerationIsReproducible() {
        ChatModel a = ChatModel.fromGguf(MiniChatGgufBuilder.buildChatDemoModel());
        ChatModel b = ChatModel.fromGguf(MiniChatGgufBuilder.buildChatDemoModel());
        ChatGenerationOptions options = ChatGenerationOptions.quality(a.tokenizer(), ChatSamplingRngMode.LEGACY);
        ChatGenerator genA = new ChatGenerator(a, options, ChatHistoryMode.LEGACY);
        ChatGenerator genB = new ChatGenerator(b, options, ChatHistoryMode.LEGACY);
        String first = genA.generate("Hello", ChatTemplate.PLAIN).text();
        String second = genB.generate("Hello", ChatTemplate.PLAIN).text();
        assertEquals(first, second);
        assertTrue(first.length() > 1);
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void defaultQualityOptionsUseSecureRng() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildChatDemoModel());
        ChatGenerationOptions options = ChatGenerationOptions.quality(model.tokenizer());
        assertEquals(ChatSamplingRngMode.SECURE, options.rngMode());
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void secureRngGeneratesNonEmptyCompletion() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildChatDemoModel());
        ChatGenerator generator = new ChatGenerator(model, ChatGenerationOptions.quality(model.tokenizer()), ChatHistoryMode.LEGACY);
        ChatGenerationResult result = generator.generate("Hello", ChatTemplate.PLAIN);
        assertTrue(result.text().length() > 1, result::text);
    }

    @Test
    void llama3SessionClosesEachTurnWithEot() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel());
        ChatGenerator generator = new ChatGenerator(model, ChatGenerationOptions.greedy(model.tokenizer(), 8), ChatHistoryMode.LEGACY);
        generator.continueConversation("Hello", ChatTemplate.LLAMA3);
        int[] session = generator.sessionTokenIdsForTests();
        assertEquals(model.tokenizer().eosId(), session[session.length - 1]);
    }

    @Test
    void qwen2SessionClosesEachTurnWithEot() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel());
        ChatGenerator generator = new ChatGenerator(model, ChatGenerationOptions.greedy(model.tokenizer(), 8), ChatHistoryMode.LEGACY);
        generator.continueConversation("Hello", ChatTemplate.QWEN2);
        int[] session = generator.sessionTokenIdsForTests();
        int eos = model.tokenizer().eosId();
        assertTrue(
                lastIndexOf(session, eos) >= session.length - 8,
                "session should end with eos (qwen adds newline after)");
    }

    private static int lastIndexOf(int[] array, int value) {
        for (int i = array.length - 1; i >= 0; i--) {
            if (array[i] == value) {
                return i;
            }
        }
        return -1;
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void multiTurnReusesKvPrefix() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildChatDemoModel());
        ChatGenerationOptions options = ChatGenerationOptions.quality(model.tokenizer(), ChatSamplingRngMode.LEGACY);
        ChatGenerator generator = new ChatGenerator(model, options, ChatHistoryMode.LEGACY);

        ChatGenerationResult first = generator.continueConversation("Hello", ChatTemplate.PLAIN);
        assertTrue(first.text().length() > 1, first::text);
        assertEquals(0, first.prefixReuseTokens());

        ChatGenerationResult second = generator.continueConversation("Hello", ChatTemplate.PLAIN);
        assertTrue(second.prefixReuseTokens() > 0, "second turn should reuse prior KV prefix");
        assertTrue(second.text().length() > 1, second::text);
    }

    @Test
    void llamaHistoryModeDeltaTurnDoesNotReportPrefixReuse() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel());
        ChatGenerationOptions options = ChatGenerationOptions.greedy(model.tokenizer(), 4);
        ChatGenerator generator = new ChatGenerator(model, options, ChatHistoryMode.LLAMA);

        ChatGenerationResult first = generator.continueConversation("Hello", ChatTemplate.LLAMA3);
        assertEquals(0, first.prefixReuseTokens());
        int sessionAfterFirst = generator.sessionTokenIds().length;

        ChatGenerationResult second = generator.continueConversation("Again", ChatTemplate.LLAMA3);
        assertEquals(0, second.prefixReuseTokens(), "llama mode tokenizes delta only");
        assertTrue(
                generator.sessionTokenIds().length > sessionAfterFirst,
                "second turn should extend closed template token ids");
        assertEquals(4, generator.messages().size());
    }

    @Test
    void prefillSliceLengthsMatchTinygradChunkBoundaries() {
        assertArrayEquals(new int[] {4, 4}, ChatGenerator.prefillSliceLengths(8, 0, 4));
        assertArrayEquals(new int[] {4, 4, 1}, ChatGenerator.prefillSliceLengths(9, 0, 4));
        assertArrayEquals(new int[] {4}, ChatGenerator.prefillSliceLengths(4, 0, 4));
        assertArrayEquals(new int[] {2, 2}, ChatGenerator.prefillSliceLengths(4, 0, 2));
    }

    @Test
    void forwardPrefillFlagsMatchTinygradChunkedPrefillTest() {
        assertArrayEquals(
                new boolean[] {true, true, false, false},
                ChatGenerator.forwardPrefillFlags(8, 4, 3));
        assertArrayEquals(
                new boolean[] {true, true, true, false, false},
                ChatGenerator.forwardPrefillFlags(9, 4, 3));
        assertArrayEquals(
                new boolean[] {true, false, false},
                ChatGenerator.forwardPrefillFlags(4, 4, 3));
    }

    @Test
    void chunkedPrefillAdvancesKvBySliceLengths() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        int[] prompt = {1, 2, 1, 2, 1, 2, 1, 2};
        int[] slices = ChatGenerator.prefillSliceLengths(prompt.length, 0, 4);
        model.resetCache();
        int pos = 0;
        for (int slice : slices) {
            model.forward(Arrays.copyOfRange(prompt, pos, pos + slice));
            pos += slice;
            assertEquals(pos, model.kvLength());
        }
        assertArrayEquals(new int[] {4, 4}, slices);
    }

    @Test
    void kvCacheInvalidatesOnDivergentPrefix() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        ChatGenerationOptions options = greedyWithPrefillChunk(model.tokenizer(), 1, 32);
        ChatGenerator generator = new ChatGenerator(model, options, ChatHistoryMode.LEGACY);

        generator.generateWithKvReuse(new int[] {1, 2, 1, 2, 1});
        assertTrue(model.kvLength() > 0);

        ChatGenerationResult divergent = generator.generateWithKvReuse(new int[] {2, 1, 2});
        assertEquals(0, divergent.prefixReuseTokens(), "divergent prompt must reset KV prefix reuse");
    }

    @Test
    void kvCacheReusesExtendedPrefix() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        ChatGenerationOptions options = greedyWithPrefillChunk(model.tokenizer(), 2, 32);
        ChatGenerator generator = new ChatGenerator(model, options, ChatHistoryMode.LEGACY);

        int[] prompt = {1, 2, 1, 2, 1};
        ChatGenerationResult first = generator.generateWithKvReuse(prompt);
        assertEquals(0, first.prefixReuseTokens());

        int[] extended = concat(prompt, first.forwardedTokenIds(), new int[] {2, 1, 2});
        ChatGenerationResult second = generator.generateWithKvReuse(extended);
        assertEquals(prompt.length + first.forwardedTokenIds().length, second.prefixReuseTokens());
    }

    @Test
    void kvCacheResumeMatchesFreshGeneration() {
        ChatModel warmModel = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        ChatModel freshModel = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        ChatGenerationOptions options = greedyWithPrefillChunk(warmModel.tokenizer(), 2, 32);

        int[] prompt = {1, 2, 1, 2, 1};
        ChatGenerator warm = new ChatGenerator(warmModel, options, ChatHistoryMode.LEGACY);
        ChatGenerationResult partial = warm.generateWithKvReuse(prompt);

        int[] extended = concat(prompt, partial.forwardedTokenIds(), new int[] {2, 1, 2});
        ChatGenerationResult resumed = warm.generateWithKvReuse(extended);

        ChatGenerator fresh = new ChatGenerator(freshModel, options, ChatHistoryMode.LEGACY);
        ChatGenerationResult fromScratch = fresh.generateWithKvReuse(extended);

        assertArrayEquals(fromScratch.generatedTokenIds(), resumed.generatedTokenIds());
        assertEquals(fromScratch.text(), resumed.text());
    }
}
