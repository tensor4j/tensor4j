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

import java.util.Random;

/**
 * Token selection from logits — argmax, or tinygrad-style sampling
 * ({@code extra/models/llama.py} alpha + top-k/top-p, {@code apps/llm.py} Gumbel-max).
 */
public final class ChatSampler {

    private ChatSampler() {
    }

    public static int argmax(float[] logits) {
        if (logits.length == 0) {
            throw new IllegalArgumentException("empty logits");
        }
        int best = 0;
        float bestValue = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < logits.length; i++) {
            if (logits[i] > bestValue) {
                bestValue = logits[i];
                best = i;
            }
        }
        return best;
    }

    public static int sample(float[] logits, ChatGenerationOptions options, int tokensGenerated, Random rng) {
        return sample(logits, options, tokensGenerated, null, new ChatSamplingRng(rng));
    }

    public static int sample(
            float[] logits,
            ChatGenerationOptions options,
            int tokensGenerated,
            ChatSamplerState state,
            Random rng) {
        return sample(logits, options, tokensGenerated, state, new ChatSamplingRng(rng));
    }

    public static int sample(
            float[] logits,
            ChatGenerationOptions options,
            int tokensGenerated,
            ChatSamplerState state,
            ChatSamplingRng samplingRng) {
        float[] work = logits.clone();
        sanitizeNaNs(work);
        if (tokensGenerated < options.minNewTokens()) {
            maskToken(work, options.bosId());
            maskToken(work, options.eosId());
        }
        if (state != null) {
            state.applyAlphaPenalties(work, options.alphaFrequency(), options.alphaPresence());
        }
        if (options.mode() == ChatGenerationMode.GREEDY || options.temperature() < 1e-6f) {
            return argmax(work);
        }
        applyTopK(work, options.topK());
        if (options.gumbelMax()) {
            return sampleGumbelMax(work, options.temperature(), samplingRng);
        }
        float[] probs = softmaxTemperature(work, options.temperature());
        applyTopP(probs, options.topP());
        return sampleMultinomial(probs, samplingRng);
    }

    private static void sanitizeNaNs(float[] logits) {
        for (int i = 0; i < logits.length; i++) {
            if (Float.isNaN(logits[i])) {
                logits[i] = Float.NEGATIVE_INFINITY;
            }
        }
    }

    private static void maskToken(float[] logits, int tokenId) {
        if (tokenId >= 0 && tokenId < logits.length) {
            logits[tokenId] = Float.NEGATIVE_INFINITY;
        }
    }

    private static void applyTopK(float[] logits, int topK) {
        if (topK <= 0 || topK >= logits.length) {
            return;
        }
        float threshold = kthLargest(logits, topK);
        for (int i = 0; i < logits.length; i++) {
            if (logits[i] < threshold) {
                logits[i] = Float.NEGATIVE_INFINITY;
            }
        }
    }

    private static float kthLargest(float[] values, int k) {
        float[] copy = values.clone();
        int target = Math.min(k, copy.length) - 1;
        for (int i = 0; i <= target; i++) {
            int best = i;
            for (int j = i + 1; j < copy.length; j++) {
                if (copy[j] > copy[best]) {
                    best = j;
                }
            }
            float tmp = copy[i];
            copy[i] = copy[best];
            copy[best] = tmp;
        }
        return copy[target];
    }

    private static int sampleGumbelMax(float[] logits, float temperature, ChatSamplingRng rng) {
        float temp = Math.max(temperature, (float) ChatSamplingRng.MIN_UNIFORM);
        int best = 0;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < logits.length; i++) {
            // tinygrad apps/llm.py: Tensor.rand_like(logits) draws noise for every vocab index
            double gumbel = rng.nextStandardGumbel();
            if (logits[i] == Float.NEGATIVE_INFINITY) {
                continue;
            }
            double score = logits[i] / temp + gumbel;
            if (score > bestScore) {
                bestScore = score;
                best = i;
            }
        }
        return best;
    }

    private static float[] softmaxTemperature(float[] logits, float temperature) {
        float max = Float.NEGATIVE_INFINITY;
        for (float logit : logits) {
            if (logit > max) {
                max = logit;
            }
        }
        float sum = 0f;
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            if (logits[i] == Float.NEGATIVE_INFINITY) {
                probs[i] = 0f;
                continue;
            }
            probs[i] = (float) Math.exp((logits[i] - max) / temperature);
            sum += probs[i];
        }
        if (sum <= 0f) {
            probs[argmax(logits)] = 1f;
            return probs;
        }
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }
        return probs;
    }

    private static void applyTopP(float[] probs, float topP) {
        if (topP >= 1f) {
            return;
        }
        int[] order = sortedIndicesDescending(probs);
        float cumulative = 0f;
        int keep = probs.length;
        for (int i = 0; i < order.length; i++) {
            cumulative += probs[order[i]];
            if (cumulative > topP) {
                keep = i + 1;
                break;
            }
        }
        boolean[] allowed = new boolean[probs.length];
        for (int i = 0; i < keep; i++) {
            allowed[order[i]] = true;
        }
        float sum = 0f;
        for (int i = 0; i < probs.length; i++) {
            if (!allowed[i]) {
                probs[i] = 0f;
            }
            sum += probs[i];
        }
        if (sum > 0f) {
            for (int i = 0; i < probs.length; i++) {
                probs[i] /= sum;
            }
        }
    }

    private static int[] sortedIndicesDescending(float[] probs) {
        int[] order = new int[probs.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        for (int i = 0; i < order.length; i++) {
            int best = i;
            for (int j = i + 1; j < order.length; j++) {
                if (probs[order[j]] > probs[order[best]]) {
                    best = j;
                }
            }
            int tmp = order[i];
            order[i] = order[best];
            order[best] = tmp;
        }
        return order;
    }

    private static int sampleMultinomial(float[] probs, ChatSamplingRng rng) {
        return multinomialAtRoll(probs, rng.nextMultinomialRoll());
    }

    /**
     * Inverse-CDF pick from a 1d probability vector — tinygrad {@code Tensor.multinomial()}.
     * {@code roll} is in [0,1) (same shape semantics as {@code ChatSamplingRng#nextMultinomialRoll()}).
     */
    public static int multinomialAtRoll(float[] probs, double roll) {
        double cumulative = 0d;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (roll <= cumulative) {
                return i;
            }
        }
        return argmax(probs);
    }

    /** Top-k mask on 1d logits (tinygrad {@code sample()} k-loop, applied before softmax in llama.py). */
    static float[] topKFilteredLogits(float[] logits, int topK) {
        float[] work = logits.clone();
        applyTopK(work, topK);
        return work;
    }

    /** Softmax(logits / temperature) returning 1d probs summing to 1. */
    static float[] temperatureSoftmax(float[] logits, float temperature) {
        return softmaxTemperature(logits, temperature);
    }

    /** Nucleus top-p filter on 1d probs, renormalized — tinygrad approximate top-p on top-k mass. */
    static float[] topPNucleusProbs(float[] probs, float topP) {
        float[] work = probs.clone();
        applyTopP(work, topP);
        return work;
    }

    /** Alpha-adjusted logits — tinygrad {@code logits - counter*af - (counter>0)*ap}. */
    static float[] alphaAdjustedLogits(float[] logits, ChatSamplerState state, float alphaF, float alphaP) {
        float[] work = logits.clone();
        if (state != null) {
            state.applyAlphaPenalties(work, alphaF, alphaP);
        }
        return work;
    }
}
