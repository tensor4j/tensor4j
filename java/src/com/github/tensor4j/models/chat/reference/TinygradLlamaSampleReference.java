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

import com.github.tensor4j.models.chat.ChatSampler;

/**
 * Golden reference for tinygrad {@code extra/models/llama.py} {@code sample()} on 1d logits.
 *
 * <p>Pipeline: temp check → alpha penalties → NaN mask → softmax(logits/temp) → iterative top-k on
 * probs → approximate top-p on kept mass → inverse-CDF multinomial.
 *
 * <p>Input shape: {@code logits [vocab]}. Output: scalar token id.
 */
public final class TinygradLlamaSampleReference {

    private final int[] alphaCounter;

    public TinygradLlamaSampleReference(int vocabSize) {
        this.alphaCounter = new int[vocabSize];
    }

    public void record(int tokenId) {
        if (tokenId >= 0 && tokenId < alphaCounter.length) {
            alphaCounter[tokenId]++;
        }
    }

    public void reset() {
        for (int i = 0; i < alphaCounter.length; i++) {
            alphaCounter[i] = 0;
        }
    }

    public int sample(
            float[] logits,
            float temperature,
            int topK,
            float topP,
            float alphaFrequency,
            float alphaPresence,
            double multinomialRoll) {
        if (temperature < 1e-6f) {
            return ChatSampler.argmax(logits);
        }

        float[] work = logits.clone();
        for (int i = 0; i < work.length; i++) {
            int count = i < alphaCounter.length ? alphaCounter[i] : 0;
            if (count > 0) {
                work[i] -= count * alphaFrequency + alphaPresence;
            }
            if (Float.isNaN(work[i])) {
                work[i] = Float.NEGATIVE_INFINITY;
            }
        }

        float[] probs = softmaxTemperature(work, temperature);
        if (topK > 0 && topK < probs.length) {
            return sampleTopKTopP(probs, topK, topP, multinomialRoll);
        }
        float[] nucleus = topPNucleus(probs, topP);
        return ChatSampler.multinomialAtRoll(nucleus, multinomialRoll);
    }

    private static int sampleTopKTopP(float[] probs, int topK, float topP, double roll) {
        int vocab = probs.length;
        float[] topProbs = new float[topK];
        int[] topIndices = new int[topK];
        float[] remaining = probs.clone();

        for (int slot = 0; slot < topK; slot++) {
            int argmax = argmaxIndex(remaining);
            topProbs[slot] = remaining[argmax];
            topIndices[slot] = argmax;
            remaining[argmax] = 0f;
        }

        float tailMass = 0f;
        for (float p : remaining) {
            tailMass += p;
        }

        float[] keptProbs = new float[topK];
        int[] keptIndices = new int[topK];
        int kept = 0;
        float cumulative = tailMass;
        for (int i = topK - 1; i >= 0; i--) {
            cumulative += topProbs[i];
            if (cumulative >= 1f - topP) {
                keptProbs[kept] = topProbs[i];
                keptIndices[kept] = topIndices[i];
                kept++;
            }
        }

        if (kept == 0) {
            keptProbs[0] = topProbs[topK - 1];
            keptIndices[0] = topIndices[topK - 1];
            kept = 1;
        }

        float sum = 0f;
        for (int i = 0; i < kept; i++) {
            sum += keptProbs[i];
        }
        double target = roll * sum;
        double cumulativeRoll = 0d;
        for (int i = 0; i < kept; i++) {
            cumulativeRoll += keptProbs[i];
            if (target <= cumulativeRoll) {
                return keptIndices[i];
            }
        }
        return keptIndices[kept - 1];
    }

    private static float[] topPNucleus(float[] probs, float topP) {
        if (topP >= 1f) {
            return probs.clone();
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
        float[] out = new float[probs.length];
        float sum = 0f;
        for (int i = 0; i < keep; i++) {
            out[order[i]] = probs[order[i]];
            sum += out[order[i]];
        }
        if (sum > 0f) {
            for (int i = 0; i < out.length; i++) {
                out[i] /= sum;
            }
        }
        return out;
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
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }
        return probs;
    }

    private static int argmaxIndex(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
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
}
