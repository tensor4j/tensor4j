/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.chat.reference;

import com.github.tensor4j.models.chat.ChatGenerationMode;
import com.github.tensor4j.models.chat.ChatGenerationOptions;
import com.github.tensor4j.models.chat.ChatSampler;
import com.github.tensor4j.models.chat.ChatSamplerState;
import com.github.tensor4j.models.chat.ChatSamplingRng;
import com.github.tensor4j.models.chat.ChatSamplingRngMode;

/** Checked-in golden vectors for tinygrad sampling parity (logits [vocab] → token id). */
public final class TinygradSamplingGoldenCases {

    private TinygradSamplingGoldenCases() {
    }

    public static TinygradSamplingGoldenCase[] all() {
        return new TinygradSamplingGoldenCase[] {
                gumbel("gumbel_v3_seed42", new float[] {1f, 2f, 0.5f}, 0.7f, 42L),
                gumbel("gumbel_v5_seed7", new float[] {1f, 5f, 4f, 0f, -10f}, 1f, 7L),
                gumbel("gumbel_v4_low_temp", new float[] {1f, 5f, 2f, 0f}, 0.01f, 42L),
                llama("llama_v4_roll_0_10", new float[] {0f, 1f, 2f, 3f}, 1f, 0.10),
                llama("llama_v5_topk2_topp09_roll_0_99", new float[] {1f, 5f, 4f, 0f, -10f}, 1f, 2, 0.9f, 0.99),
                llamaAlpha(
                        "llama_alpha_fp_roll_0_50",
                        new float[] {0f, 10f, 10f},
                        1f,
                        0.1f,
                        0.5f,
                        new int[] {1, 1},
                        0.50),
        };
    }

    public static int sampleWithChatSampler(TinygradSamplingGoldenCase golden) {
        ChatGenerationOptions options = new ChatGenerationOptions(
                ChatGenerationMode.QUALITY,
                golden.temperature(),
                golden.topP(),
                golden.topK(),
                0,
                8,
                golden.seed(),
                -1,
                -1,
                -1,
                golden.alphaFrequency(),
                golden.alphaPresence(),
                golden.gumbelMax(),
                32,
                ChatSamplingRngMode.LEGACY);
        ChatSamplerState state = null;
        if (golden.alphaCounts() != null) {
            state = new ChatSamplerState(golden.logits().length);
            for (int tokenId : golden.alphaCounts()) {
                state.record(tokenId);
            }
        }
        ChatSamplingRng rng = golden.gumbelMax()
                ? new ChatSamplingRng(golden.seed())
                : ChatSamplingRng.fixedMultinomialRoll(golden.multinomialRoll());
        return ChatSampler.sample(golden.logits(), options, 0, state, rng);
    }

    public static int sampleWithReference(TinygradSamplingGoldenCase golden) {
        if (golden.gumbelMax()) {
            return TinygradGumbelSampleReference.sample(golden.logits(), golden.temperature(), golden.seed());
        }
        TinygradLlamaSampleReference ref = new TinygradLlamaSampleReference(golden.logits().length);
        if (golden.alphaCounts() != null) {
            for (int tokenId : golden.alphaCounts()) {
                ref.record(tokenId);
            }
        }
        return ref.sample(
                golden.logits(),
                golden.temperature(),
                golden.topK(),
                golden.topP(),
                golden.alphaFrequency(),
                golden.alphaPresence(),
                golden.multinomialRoll());
    }

    private static TinygradSamplingGoldenCase gumbel(String name, float[] logits, float temp, long seed) {
        int expected = TinygradGumbelSampleReference.sample(logits, temp, seed);
        return new TinygradSamplingGoldenCase(
                name, logits, temp, seed, 0d, 0, 1f, 0f, 0f, null, true, expected);
    }

    private static TinygradSamplingGoldenCase llama(
            String name, float[] logits, float temp, double roll) {
        return llama(name, logits, temp, 0, 1f, roll);
    }

    private static TinygradSamplingGoldenCase llama(
            String name, float[] logits, float temp, int topK, float topP, double roll) {
        int expected = new TinygradLlamaSampleReference(logits.length)
                .sample(logits, temp, topK, topP, 0f, 0f, roll);
        return new TinygradSamplingGoldenCase(
                name, logits, temp, 0L, roll, topK, topP, 0f, 0f, null, false, expected);
    }

    private static TinygradSamplingGoldenCase llamaAlpha(
            String name,
            float[] logits,
            float temp,
            float alphaF,
            float alphaP,
            int[] counts,
            double roll) {
        TinygradLlamaSampleReference ref = new TinygradLlamaSampleReference(logits.length);
        for (int tokenId : counts) {
            ref.record(tokenId);
        }
        int expected = ref.sample(logits, temp, 0, 1f, alphaF, alphaP, roll);
        return new TinygradSamplingGoldenCase(
                name, logits, temp, 0L, roll, 0, 1f, alphaF, alphaP, counts, false, expected);
    }
}
