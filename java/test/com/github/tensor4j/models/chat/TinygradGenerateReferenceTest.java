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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.reference.TinygradGenerateReference;
import org.junit.jupiter.api.Test;

class TinygradGenerateReferenceTest {

    @Test
    void getStartPosMatchesChatGenerator() {
        int[] tokens = {1, 2, 9};
        int[] cached = {1, 2, 3};
        assertEquals(
                ChatGenerator.sharedPrefixLength(tokens, cached),
                TinygradGenerateReference.getStartPos(tokens, cached));
    }

    @Test
    void eosStopRespectsMinNewTokens() {
        assertFalse(TinygradGenerateReference.shouldStop(10, 10, 0, 2));
        assertFalse(TinygradGenerateReference.shouldStop(10, 10, 1, 2));
        assertTrue(TinygradGenerateReference.shouldStop(10, 10, 2, 2));
    }

    @Test
    void skipDecodeCoversBosAndEos() {
        assertTrue(TinygradGenerateReference.skipDecode(0, 0, 99));
        assertTrue(TinygradGenerateReference.skipDecode(99, 0, 99));
        assertFalse(TinygradGenerateReference.skipDecode(7, 0, 99));
    }
}
