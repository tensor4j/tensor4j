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

/** Token selection from logits — argmax or temperature + top-k/top-p sampling. */
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
        float[] work = logits.clone();
        if (tokensGenerated < options.minNewTokens()) {
            maskToken(work, options.bosId());
            maskToken(work, options.eosId());
        }
        if (options.mode() == ChatGenerationMode.GREEDY || options.temperature() < 1e-6f) {
            return argmax(work);
        }
        applyTopK(work, options.topK());
        float[] probs = softmaxTemperature(work, options.temperature());
        applyTopP(probs, options.topP());
        return sampleMultinomial(probs, rng);
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

    private static int sampleMultinomial(float[] probs, Random rng) {
        double roll = rng.nextDouble();
        double cumulative = 0d;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (roll <= cumulative) {
                return i;
            }
        }
        return argmax(probs);
    }
}
