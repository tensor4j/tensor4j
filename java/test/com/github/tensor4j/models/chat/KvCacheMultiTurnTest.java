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
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.runtime.gguf.GgufTensorSource;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Multi-turn KV invariants targeting progressive corruption (wrong write index, stale
 * {@code n_past}, RoPE mismatch, attention not seeing full history).
 */
class KvCacheMultiTurnTest {

    private static final String[] FIVE_TURN_QUESTIONS = {
        "Capital of England?",
        "Opposite of dark?",
        "Four multiplied by five?",
        "Benefits of playing golf?",
        "Why shorter answers?",
    };

    @Test
    void eachForwardIncrementsKvByExactBatchSize() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        model.resetCache();
        assertEquals(0, model.kvLength());

        int[][] batches = {{1, 2}, {1}, {2, 1, 2}};
        int total = 0;
        for (int[] batch : batches) {
            int before = model.kvLength();
            model.forward(batch);
            total += batch.length;
            assertEquals(
                    before + batch.length,
                    model.kvLength(),
                    "KV write index must advance by batch size (no overwrite/restart)");
            assertEquals(total, model.kvLength());
        }
    }

    @Test
    void decodeLogitsDependOnFullCachedHistory() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        int[] history = {1, 2, 1, 2, 1, 2, 1, 2};
        int[] decode = {1};

        model.resetCache();
        model.forward(history);
        float[] withHistory = model.forward(decode);

        model.resetCache();
        model.forward(new int[] {2, 1});
        float[] tailOnly = model.forward(decode);

        assertFalse(
                arraysNearEqual(withHistory, tailOnly, 1e-6f),
                "attention must read entire KV — not just the latest prompt chunk");
    }

    @Test
    void llamaIncrementalDeltaPrefillMatchesColdFullTemplateLogits() {
        GgufTensorSource weights = MiniChatGgufBuilder.buildLlama3TemplateModel();
        ChatModel warm = ChatModel.fromGguf(weights);
        ChatTokenizer tokenizer = warm.tokenizer();
        ChatGenerator generator = new ChatGenerator(
                warm, ChatGenerationOptions.greedy(tokenizer, 2), ChatHistoryMode.LLAMA);

        generator.continueConversation("Hi", ChatTemplate.LLAMA3);
        generator.continueConversation("Yo", ChatTemplate.LLAMA3);

        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);
        List<ChatMessage> planned = new ArrayList<>(generator.messages());
        planned.add(new ChatMessage("user", "Sup"));
        int prev = generator.templatePrevTokens();
        int[] full = applier.tokenIds(tokenizer, planned, true);
        int[] delta = applier.tokenDeltaSince(tokenizer, planned, true, prev);

        assertTrue(delta.length > 0, "turn-3 must have non-empty delta");
        float[] warmLogits = warm.forward(delta);

        ChatModel cold = ChatModel.fromGguf(weights);
        cold.resetCache();
        float[] coldLogits = cold.forward(full);

        if (full.length <= warm.config().nCtx()) {
            assertArrayEquals(
                    coldLogits,
                    warmLogits,
                    1e-4f,
                    "delta prefill on warm KV must match cold full template (llama.cpp n_past parity)");
        }
    }

    @Test
    void llamaFiveTurnSessionGrowsAndKvTracksTemplateWithinContext() {
        GgufTensorSource weights = MiniChatGgufBuilder.buildLlama3TemplateModel();
        ChatModel model = ChatModel.fromGguf(weights);
        ChatGenerator generator = new ChatGenerator(
                model, ChatGenerationOptions.greedy(model.tokenizer(), 4), ChatHistoryMode.LLAMA);

        int prevSession = 0;
        int prevKv = 0;
        for (String question : FIVE_TURN_QUESTIONS) {
            generator.continueConversation(question, ChatTemplate.LLAMA3);
            int session = generator.sessionTokenIds().length;
            int kv = model.kvLength();
            int closed = generator.templatePrevTokens();

            assertTrue(session > prevSession, "session token buffer must grow each turn");
            assertTrue(kv >= prevKv, "KV must not shrink between turns");
            assertEquals(session, closed, "closed template must match session ids");

            if (closed <= model.config().nCtx()) {
                assertEquals(
                        closed,
                        kv,
                        "kv_len must equal total tokens seen when within n_ctx (turn: " + question + ")");
            } else {
                assertEquals(
                        model.config().nCtx(),
                        kv,
                        "ring KV caps at n_ctx — RoPE absolute index may diverge (known limitation)");
            }
            prevSession = session;
            prevKv = kv;
        }
    }

    @Test
    void legacyFiveTurnKvMatchesCachedTokenCountEachTurn() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        ChatGenerator generator = new ChatGenerator(
                model, ChatGenerationOptions.greedy(model.tokenizer(), 1), ChatHistoryMode.LEGACY);

        for (int turn = 0; turn < 5; turn++) {
            generator.continueConversation("q" + turn, ChatTemplate.PLAIN);
            int cached = generator.cachedTokenIds().length;
            assertEquals(
                    cached,
                    model.kvLength(),
                    "tinygrad _cached_tokens length must track n_past each turn (turn " + turn + ")");
        }
    }

    @Test
    void legacyExtendedPromptReusesKvPrefix() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        ChatGenerationOptions options = ChatGenerationOptions.greedy(model.tokenizer(), 2);
        ChatGenerator generator = new ChatGenerator(model, options, ChatHistoryMode.LEGACY);

        int[] prompt = {1, 2, 1, 2, 1};
        ChatGenerationResult first = generator.generateWithKvReuse(prompt);
        int[] extended = concat(prompt, first.forwardedTokenIds(), new int[] {2, 1, 2});
        int kvAfterFirst = model.kvLength();
        int startPos = ChatGenerator.sharedPrefixLength(extended, generator.cachedTokenIds());
        assertEquals(kvAfterFirst, startPos, "n_past must equal get_start_pos before extended prefill");

        ChatGenerationResult second = generator.generateWithKvReuse(extended);
        assertTrue(
                second.prefixReuseTokens() > 0,
                "extended prompt must reuse KV prefix (tinygrad get_start_pos > 0)");
    }

    @Test
    void legacyExtendedPromptResumeMatchesColdWhenPrefixAligned() {
        GgufTensorSource weights = MiniChatGgufBuilder.buildIdentityModel();
        ChatModel warm = ChatModel.fromGguf(weights);
        ChatGenerationOptions options = ChatGenerationOptions.greedy(warm.tokenizer(), 2);
        ChatGenerator generator = new ChatGenerator(warm, options, ChatHistoryMode.LEGACY);

        int[] prompt = {1, 2, 1, 2, 1};
        ChatGenerationResult partial = generator.generateWithKvReuse(prompt);
        int[] extended = concat(prompt, partial.forwardedTokenIds(), new int[] {2, 1, 2});

        int startPos = ChatGenerator.sharedPrefixLength(extended, generator.cachedTokenIds());
        assertTrue(startPos > 0);
        assertEquals(warm.kvLength(), startPos);

        float[] warmLogits = ChatGenerator.prefillChunked(warm, extended, startPos, 2);
        ChatModel cold = ChatModel.fromGguf(weights);
        cold.resetCache();
        float[] coldLogits = cold.forward(extended);
        assertArrayEquals(coldLogits, warmLogits, 1e-4f);
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
    void writingAtWrongStartPosWouldChangeLogits() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        int[] prompt = {1, 2, 1, 2, 1, 2};

        model.resetCache();
        model.forward(Arrays.copyOfRange(prompt, 0, 3));
        float[] correct = ChatGenerator.prefillChunked(model, prompt, 3, 2);

        ChatModel wrong = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        wrong.resetCache();
        wrong.forward(Arrays.copyOfRange(prompt, 0, 3));
        float[] offByOne = ChatGenerator.prefillChunked(wrong, prompt, 2, 2);

        assertFalse(
                arraysNearEqual(correct, offByOne, 1e-4f),
                "start_pos off-by-one must change logits — guards cache position regression");
    }

    private static boolean arraysNearEqual(float[] a, float[] b, float eps) {
        if (a.length != b.length) {
            return false;
        }
        for (int i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - b[i]) > eps) {
                return false;
            }
        }
        return true;
    }
}
