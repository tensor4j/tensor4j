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
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.security.SecureRandom;

import org.junit.jupiter.api.Test;

class ChatSamplingRngTest {

    @Test
    void standardGumbelFromKnownUniform() {
        double g = ChatSamplingRng.standardGumbelFromUniform(0.5);
        assertEquals(-Math.log(-Math.log(0.5)), g, 1e-9);
    }

    @Test
    void clampedUniformMatchesTinygradMinimum() {
        ChatSamplingRng rng = new ChatSamplingRng(0L);
        for (int i = 0; i < 100; i++) {
            assertTrue(rng.nextUniformClamped() >= ChatSamplingRng.MIN_UNIFORM);
        }
    }

    @Test
    void gumbelMaxScoreIsReproducibleWithSeed() {
        ChatSamplingRng a = new ChatSamplingRng(42L);
        ChatSamplingRng b = new ChatSamplingRng(42L);
        assertEquals(a.gumbelMaxScore(3.5f, 0.7f), b.gumbelMaxScore(3.5f, 0.7f), 1e-9);
        assertEquals(a.gumbelMaxScore(1.0f, 0.7f), b.gumbelMaxScore(1.0f, 0.7f), 1e-9);
    }

    @Test
    void secureModeUsesSecureRandomSource() {
        ChatSamplingRng rng = ChatSamplingRng.create(ChatSamplingRngMode.SECURE, 0L);
        assertEquals(ChatSamplingRngMode.SECURE, rng.mode());
        assertTrue(rng.source() instanceof SecureRandom);
    }

    @Test
    void legacyModeUsesSeededRandom() {
        ChatSamplingRng a = ChatSamplingRng.create(ChatSamplingRngMode.LEGACY, 42L);
        ChatSamplingRng b = ChatSamplingRng.create(ChatSamplingRngMode.LEGACY, 42L);
        assertEquals(a.nextMultinomialRoll(), b.nextMultinomialRoll(), 1e-9);
    }

    @Test
    void rngModeFromString() {
        assertEquals(ChatSamplingRngMode.SECURE, ChatSamplingRngMode.fromString(null));
        assertEquals(ChatSamplingRngMode.LEGACY, ChatSamplingRngMode.fromString("legacy"));
        assertEquals(ChatSamplingRngMode.SECURE, ChatSamplingRngMode.fromString("secure"));
    }

    @Test
    void multinomialRollsDifferAcrossSeeds() {
        ChatSamplingRng a = new ChatSamplingRng(1L);
        ChatSamplingRng b = new ChatSamplingRng(2L);
        double rollA = a.nextMultinomialRoll();
        double rollB = b.nextMultinomialRoll();
        assertTrue(rollA >= 0.0 && rollA < 1.0);
        assertTrue(rollB >= 0.0 && rollB < 1.0);
    }
}
