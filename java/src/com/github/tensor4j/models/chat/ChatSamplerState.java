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

/** Per-session token counts for tinygrad-style alpha frequency/presence penalties. */
public final class ChatSamplerState {

    private final int[] tokenCounts;

    public ChatSamplerState(int vocabSize) {
        if (vocabSize <= 0) {
            throw new IllegalArgumentException("vocabSize must be positive");
        }
        this.tokenCounts = new int[vocabSize];
    }

    public void record(int tokenId) {
        if (tokenId >= 0 && tokenId < tokenCounts.length) {
            tokenCounts[tokenId]++;
        }
    }

    public void reset() {
        for (int i = 0; i < tokenCounts.length; i++) {
            tokenCounts[i] = 0;
        }
    }

    /** {@code logits -= count * alpha_f + (count > 0 ? alpha_p : 0)} (tinygrad {@code sample()}). */
    public void applyAlphaPenalties(float[] logits, float alphaFrequency, float alphaPresence) {
        if (alphaFrequency == 0f && alphaPresence == 0f) {
            return;
        }
        int n = Math.min(logits.length, tokenCounts.length);
        for (int i = 0; i < n; i++) {
            int count = tokenCounts[i];
            if (count > 0) {
                logits[i] -= count * alphaFrequency + alphaPresence;
            }
        }
    }
}
