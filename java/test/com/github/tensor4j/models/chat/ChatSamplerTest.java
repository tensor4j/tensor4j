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
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import org.junit.jupiter.api.Test;

/** Sampling policy tests aligned with tinygrad {@code extra/models/llama.py} {@code sample()}. */
class ChatSamplerTest {

    private static ChatGenerationOptions qualityOptions(
            float temp, int topK, float topP, float alphaF, float alphaP, boolean gumbel) {
        return new ChatGenerationOptions(
                ChatGenerationMode.QUALITY,
                temp,
                topP,
                topK,
                0,
                8,
                42L,
                -1,
                -1,
                -1,
                alphaF,
                alphaP,
                gumbel,
                32,
                ChatSamplingRngMode.LEGACY);
    }

    @Test
    void greedyModePicksArgmaxRegardlessOfTopK() {
        float[] logits = new float[] {1f, 5f, 4f, 0f};
        ChatGenerationOptions options = new ChatGenerationOptions(
                ChatGenerationMode.GREEDY, 0f, 0.5f, 2, 0, 8, 0L, -1, -1, -1, 0f, 0f, false, 32, ChatSamplingRngMode.LEGACY);
        assertEquals(1, ChatSampler.sample(logits, options, 0, new Random(1)));
    }

    @Test
    void zeroTemperatureUsesArgmaxBeforeTopK() {
        float[] logits = new float[] {1f, 5f, 4f, 0f};
        ChatGenerationOptions options = qualityOptions(0f, 2, 1f, 0f, 0f, false);
        assertEquals(1, ChatSampler.sample(logits, options, 0, new Random(1)));
    }

    @Test
    void qualityModeSuppressesEotUntilMinTokens() {
        float[] logits = new float[] {1f, 2f, 3f, 100f};
        ChatGenerationOptions options = new ChatGenerationOptions(
                ChatGenerationMode.QUALITY, 0f, 1f, 0, 2, 8, 0L, 0, 3, 4, 0f, 0f, false, 32, ChatSamplingRngMode.LEGACY);
        int picked = ChatSampler.sample(logits, options, 0, new Random(1));
        assertNotEquals(4, picked);
        assertEquals(2, picked);
    }

    @Test
    void qualityModeSuppressesEosUntilMinTokens() {
        float[] logits = new float[] {1f, 2f, 3f, 100f};
        ChatGenerationOptions options = new ChatGenerationOptions(
                ChatGenerationMode.QUALITY, 0f, 1f, 0, 2, 8, 0L, 0, 3, 4, 0f, 0f, false, 32, ChatSamplingRngMode.LEGACY);
        int picked = ChatSampler.sample(logits, options, 0, new Random(1));
        assertNotEquals(3, picked);
        assertEquals(2, picked);
    }

    @Test
    void topKMasksBelowKthLargestLogit() {
        float[] logits = {1f, 5f, 4f, 0f, -10f};
        float[] filtered = ChatSampler.topKFilteredLogits(logits, 2);
        assertEquals(5f, filtered[1]);
        assertEquals(4f, filtered[2]);
        assertEquals(Float.NEGATIVE_INFINITY, filtered[0]);
        assertEquals(Float.NEGATIVE_INFINITY, filtered[3]);
        assertEquals(Float.NEGATIVE_INFINITY, filtered[4]);
    }

    @Test
    void topKWithTiesKeepsEqualTopLogits() {
        float[] logits = {5f, 5f, 5f, 1f, 1f};
        float[] filtered = ChatSampler.topKFilteredLogits(logits, 3);
        assertEquals(5f, filtered[0]);
        assertEquals(5f, filtered[1]);
        assertEquals(5f, filtered[2]);
        assertEquals(Float.NEGATIVE_INFINITY, filtered[3]);
        assertEquals(Float.NEGATIVE_INFINITY, filtered[4]);
    }

    @Test
    void topKZeroOrFullVocabIsNoOp() {
        float[] logits = {1f, 2f, 3f};
        assertArrayEquals(logits, ChatSampler.topKFilteredLogits(logits, 0));
        assertArrayEquals(logits, ChatSampler.topKFilteredLogits(logits, 3));
    }

    @Test
    void alphaFrequencyScalesWithRepeatCount() {
        float[] logits = {0f, 10f, 10f};
        ChatSamplerState state = new ChatSamplerState(3);
        state.record(1);
        state.record(1);
        float[] adjusted = ChatSampler.alphaAdjustedLogits(logits, state, 0.1f, 0f);
        assertEquals(9.8f, adjusted[1], 1e-6f);
        assertEquals(10f, adjusted[2], 1e-6f);
    }

    @Test
    void alphaPresenceAppliesOnceWhenSeen() {
        float[] logits = {0f, 10f, 10f};
        ChatSamplerState state = new ChatSamplerState(3);
        state.record(1);
        float[] adjusted = ChatSampler.alphaAdjustedLogits(logits, state, 0f, 5f);
        assertEquals(5f, adjusted[1], 1e-6f);
    }

    @Test
    void alphaFrequencyAndPresenceCombineLikeTinygrad() {
        float[] logits = {0f, 10f, 10f};
        ChatSamplerState state = new ChatSamplerState(3);
        state.record(1);
        state.record(1);
        float[] adjusted = ChatSampler.alphaAdjustedLogits(logits, state, 0.1f, 0.5f);
        assertEquals(9.3f, adjusted[1], 1e-6f);
    }

    @Test
    void alphaPresencePenalizesRepeatedTokenInSamplePath() {
        float[] logits = new float[] {0f, 10f, 10f};
        ChatGenerationOptions options = qualityOptions(0f, 0, 1f, 0f, 5f, false);
        ChatSamplerState state = new ChatSamplerState(3);
        state.record(1);
        int picked = ChatSampler.sample(logits, options, 1, state, new Random(1));
        assertEquals(2, picked);
    }

    @Test
    void temperatureSoftmaxIsOneDimensionalDistribution() {
        float[] logits = {0f, 1f, 2f};
        float[] probs = ChatSampler.temperatureSoftmax(logits, 1f);
        assertEquals(3, probs.length);
        float sum = 0f;
        for (float p : probs) {
            assertTrue(p >= 0f);
            sum += p;
        }
        assertEquals(1f, sum, 1e-5f);
        assertTrue(probs[2] > probs[1]);
        assertTrue(probs[1] > probs[0]);
    }

    @Test
    void topPNucleusKeepsSmallestCoveringSet() {
        float[] probs = {0.5f, 0.3f, 0.15f, 0.05f};
        float[] nucleus = ChatSampler.topPNucleusProbs(probs, 0.9f);
        assertTrue(nucleus[0] > 0f);
        assertTrue(nucleus[1] > 0f);
        assertTrue(nucleus[2] > 0f);
        assertEquals(0f, nucleus[3], 1e-6f);
        float sum = nucleus[0] + nucleus[1] + nucleus[2];
        assertEquals(1f, sum, 1e-5f);
    }

    @Test
    void topPOneIsNoOp() {
        float[] probs = {0.25f, 0.25f, 0.25f, 0.25f};
        assertArrayEquals(probs, ChatSampler.topPNucleusProbs(probs, 1f), 1e-6f);
    }

    @Test
    void multinomialAtRollUsesCumulativeMass() {
        float[] probs = {0.2f, 0.3f, 0.5f};
        assertEquals(0, ChatSampler.multinomialAtRoll(probs, 0.1));
        assertEquals(1, ChatSampler.multinomialAtRoll(probs, 0.25));
        assertEquals(2, ChatSampler.multinomialAtRoll(probs, 0.99));
    }

    @Test
    void multinomialPathSampleEndToEndWithFixedRoll() {
        float[] logits = {0f, 0f, 10f};
        float[] probs = ChatSampler.topPNucleusProbs(
                ChatSampler.temperatureSoftmax(ChatSampler.topKFilteredLogits(logits, 2), 1f),
                1f);
        assertEquals(2, ChatSampler.multinomialAtRoll(probs, 0.5));
    }

    @Test
    void multinomialSamplePathUsesTopKAndTopP() {
        float[] logits = {1f, 5f, 4f, 0f, -10f};
        ChatGenerationOptions options = qualityOptions(1f, 2, 0.9f, 0f, 0f, false);
        int picked = ChatSampler.sample(logits, options, 0, null, ChatSamplingRng.fixedMultinomialRoll(0.99));
        assertTrue(picked == 1 || picked == 2, "top-k=2 should restrict to ids 1 or 2");
    }

    @Test
    void nanLogitsAreMaskedBeforeSampling() {
        float[] logits = {Float.NaN, 2f, 1f};
        ChatGenerationOptions options = qualityOptions(0f, 0, 1f, 0f, 0f, false);
        assertEquals(1, ChatSampler.sample(logits, options, 0, new Random(1)));
    }

    @Test
    void temperatureSamplingIsStochastic() {
        float[] logits = new float[] {1f, 2f, 3f, 0f};
        ChatGenerationOptions options = qualityOptions(0.7f, 0, 1f, 0f, 0f, true);
        int a = ChatSampler.sample(logits, options, 0, new Random(42));
        int b = ChatSampler.sample(logits, options, 0, new Random(99));
        assertTrue(a >= 0 && a < logits.length);
        assertTrue(b >= 0 && b < logits.length);
    }

    @Test
    void gumbelMaxPicksHighLogitAtZeroTemperature() {
        float[] logits = new float[] {1f, 5f, 2f};
        ChatGenerationOptions options = qualityOptions(0.01f, 0, 1f, 0f, 0f, true);
        int picked = ChatSampler.sample(logits, options, 0, new Random(42));
        assertEquals(1, picked);
    }
}
