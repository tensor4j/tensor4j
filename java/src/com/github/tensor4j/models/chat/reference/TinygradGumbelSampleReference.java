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

import com.github.tensor4j.models.chat.ChatSamplingRng;
import java.util.Random;

/**
 * Golden reference for tinygrad {@code apps/llm.py} Gumbel-max decode:
 * {@code argmax(logits/temp - log(-log(uniform)))} with {@code uniform ~ U(0,1)} per vocab index.
 *
 * <p>Input shape: {@code logits [vocab]}. Output: scalar token id.
 */
public final class TinygradGumbelSampleReference {

    private TinygradGumbelSampleReference() {
    }

    public static int sample(float[] logits, float temperature, long seed) {
        Random rng = new Random(seed);
        float temp = Math.max(temperature, (float) ChatSamplingRng.MIN_UNIFORM);
        int best = 0;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < logits.length; i++) {
            double u = Math.max(rng.nextDouble(), ChatSamplingRng.MIN_UNIFORM);
            double score = logits[i] / temp + ChatSamplingRng.standardGumbelFromUniform(u);
            if (score > bestScore) {
                bestScore = score;
                best = i;
            }
        }
        return best;
    }
}
