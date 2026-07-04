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

/** Greedy token selection from logits (llama.cpp sampling argmax). */
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
}
