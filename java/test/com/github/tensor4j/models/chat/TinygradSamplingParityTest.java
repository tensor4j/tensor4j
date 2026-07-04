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

import com.github.tensor4j.models.chat.reference.TinygradGumbelSampleReference;
import com.github.tensor4j.models.chat.reference.TinygradLlamaSampleReference;
import com.github.tensor4j.models.chat.reference.TinygradSamplingGoldenCase;
import com.github.tensor4j.models.chat.reference.TinygradSamplingGoldenCases;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

/** Cross-parity: {@link ChatSampler} vs tinygrad reference samplers and golden logits+seed fixtures. */
class TinygradSamplingParityTest {

    @Test
    void gumbelChatSamplerMatchesAppsLlmReference() {
        float[][] logitsBatch = {
                {1f, 2f, 0.5f},
                {1f, 5f, 4f, 0f, -10f},
                {1f, 5f, 2f, 0f},
        };
        float[] temps = {0.7f, 1f, 0.01f};
        long[] seeds = {42L, 7L, 42L};

        for (int i = 0; i < logitsBatch.length; i++) {
            int expected = TinygradGumbelSampleReference.sample(logitsBatch[i], temps[i], seeds[i]);
            TinygradSamplingGoldenCase golden = gumbelCase(logitsBatch[i], temps[i], seeds[i], expected);
            assertEquals(expected, TinygradSamplingGoldenCases.sampleWithReference(golden));
            assertEquals(expected, TinygradSamplingGoldenCases.sampleWithChatSampler(golden));
        }
    }

    @Test
    void llamaMultinomialChatSamplerMatchesReferenceWhenTopKZero() {
        assertLlamaParity(new float[] {0f, 1f, 2f, 3f}, 1f, 0, 1f, 0f, 0f, null, 0.10);
        assertLlamaParity(new float[] {0f, 1f, 2f, 3f}, 1f, 0, 1f, 0f, 0f, null, 0.75);
    }

    @Test
    void llamaTopKTopPReferenceMatchesGoldenExpectedToken() {
        TinygradSamplingGoldenCase golden = findGolden("llama_v5_topk2_topp09_roll_0_99");
        assertEquals(golden.expectedToken(), TinygradSamplingGoldenCases.sampleWithReference(golden));
    }

    @Test
    void llamaTopKOnLogitsDiffersFromTinygradTopKOnProbs() {
        TinygradSamplingGoldenCase golden = findGolden("llama_v5_topk2_topp09_roll_0_99");
        int chat = TinygradSamplingGoldenCases.sampleWithChatSampler(golden);
        int llama = TinygradSamplingGoldenCases.sampleWithReference(golden);
        assertEquals(1, llama, "tinygrad llama.py top-k on softmax picks highest mass in top-2");
        assertTrue(chat == 1 || chat == 2, "ChatSampler top-k on logits may differ");
    }

    @Test
    void llamaAlphaPenaltiesMatchReferenceAndGoldenJson() throws Exception {
        TinygradSamplingGoldenCase golden = findGolden("llama_alpha_fp_roll_0_50");
        assertEquals(golden.expectedToken(), TinygradSamplingGoldenCases.sampleWithReference(golden));
        assertEquals(golden.expectedToken(), TinygradSamplingGoldenCases.sampleWithChatSampler(golden));
        assertEquals(golden.expectedToken(), readJsonExpected(golden.name()));
    }

    @Test
    void allGoldenCasesAgreeWithReferenceAndChatSampler() throws Exception {
        for (TinygradSamplingGoldenCase golden : TinygradSamplingGoldenCases.all()) {
            assertEquals(
                    golden.expectedToken(),
                    TinygradSamplingGoldenCases.sampleWithReference(golden),
                    golden.name() + " reference");
            assertEquals(golden.expectedToken(), readJsonExpected(golden.name()), golden.name() + " json");
            if (golden.gumbelMax() || (golden.topK() == 0 && golden.alphaCounts() == null)) {
                assertEquals(
                        golden.expectedToken(),
                        TinygradSamplingGoldenCases.sampleWithChatSampler(golden),
                        golden.name() + " chat sampler");
            } else if (golden.alphaCounts() != null) {
                assertEquals(
                        golden.expectedToken(),
                        TinygradSamplingGoldenCases.sampleWithChatSampler(golden),
                        golden.name() + " chat sampler alpha");
            }
        }
    }

    @Test
    void goldenFixturesCoverOneDimensionalLogitShapes() {
        int[] vocabSizes = new int[TinygradSamplingGoldenCases.all().length];
        for (int i = 0; i < vocabSizes.length; i++) {
            vocabSizes[i] = TinygradSamplingGoldenCases.all()[i].logits().length;
        }
        assertArrayEquals(new int[] {3, 5, 4, 4, 5, 3}, vocabSizes);
    }

    private static void assertLlamaParity(
            float[] logits,
            float temp,
            int topK,
            float topP,
            float alphaF,
            float alphaP,
            int[] counts,
            double roll) {
        TinygradLlamaSampleReference ref = new TinygradLlamaSampleReference(logits.length);
        if (counts != null) {
            for (int id : counts) {
                ref.record(id);
            }
        }
        int expected = ref.sample(logits, temp, topK, topP, alphaF, alphaP, roll);
        TinygradSamplingGoldenCase golden = llamaCase(logits, temp, topK, topP, alphaF, alphaP, counts, roll, expected);
        assertEquals(expected, TinygradSamplingGoldenCases.sampleWithChatSampler(golden));
    }

    private static TinygradSamplingGoldenCase findGolden(String name) {
        for (TinygradSamplingGoldenCase golden : TinygradSamplingGoldenCases.all()) {
            if (golden.name().equals(name)) {
                return golden;
            }
        }
        throw new IllegalArgumentException("missing golden case " + name);
    }

    private static int readJsonExpected(String name) throws Exception {
        try (InputStream in = TinygradSamplingParityTest.class.getResourceAsStream("/tinygrad-sampling-golden.json")) {
            if (in == null) {
                throw new IllegalStateException("missing tinygrad-sampling-golden.json");
            }
            String json = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            String needle = "\"name\": \"" + name + "\"";
            int idx = json.indexOf(needle);
            if (idx < 0) {
                throw new IllegalArgumentException("case not in json: " + name);
            }
            int tokenIdx = json.indexOf("\"expectedToken\": ", idx);
            int start = tokenIdx + "\"expectedToken\": ".length();
            String tail = json.substring(start).trim();
            int end = 0;
            while (end < tail.length() && (Character.isDigit(tail.charAt(end)) || tail.charAt(end) == '-')) {
                end++;
            }
            return Integer.parseInt(tail.substring(0, end));
        }
    }

    private static TinygradSamplingGoldenCase gumbelCase(float[] logits, float temp, long seed, int expected) {
        return new TinygradSamplingGoldenCase(
                "dynamic", logits, temp, seed, 0d, 0, 1f, 0f, 0f, null, true, expected);
    }

    private static TinygradSamplingGoldenCase llamaCase(
            float[] logits,
            float temp,
            int topK,
            float topP,
            float alphaF,
            float alphaP,
            int[] counts,
            double roll,
            int expected) {
        return new TinygradSamplingGoldenCase(
                "dynamic", logits, temp, 0L, roll, topK, topP, alphaF, alphaP, counts, false, expected);
    }
}
