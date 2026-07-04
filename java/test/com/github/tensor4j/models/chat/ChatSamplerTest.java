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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import org.junit.jupiter.api.Test;

class ChatSamplerTest {

    @Test
    void qualityModeSuppressesEosUntilMinTokens() {
        float[] logits = new float[] {1f, 2f, 3f, 100f};
        ChatGenerationOptions options = new ChatGenerationOptions(
                ChatGenerationMode.QUALITY, 0f, 1f, 0, 2, 8, 0L, 0, 3);
        int picked = ChatSampler.sample(logits, options, 0, new Random(1));
        assertNotEquals(3, picked);
        assertEquals(2, picked);
    }

    @Test
    void temperatureSamplingIsStochastic() {
        float[] logits = new float[] {1f, 2f, 3f, 0f};
        ChatGenerationOptions options = new ChatGenerationOptions(
                ChatGenerationMode.QUALITY, 0.7f, 1f, 0, 0, 8, 42L, -1, -1);
        int a = ChatSampler.sample(logits, options, 0, new Random(42));
        int b = ChatSampler.sample(logits, options, 0, new Random(99));
        assertTrue(a >= 0 && a < logits.length);
        assertTrue(b >= 0 && b < logits.length);
    }
}
