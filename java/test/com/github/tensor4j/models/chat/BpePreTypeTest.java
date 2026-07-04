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
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class BpePreTypeTest {

    @Test
    void mapsCommonPreStrings() {
        assertEquals(BpePreType.LLAMA3, BpePreType.fromPre("llama3"));
        assertEquals(BpePreType.QWEN2, BpePreType.fromPre("qwen2"));
        assertEquals(BpePreType.QWEN35, BpePreType.fromPre("qwen35"));
        assertEquals(BpePreType.GEMMA4, BpePreType.fromPre("gemma4"));
        assertEquals(BpePreType.GPT2, BpePreType.fromPre("gpt-2"));
        assertEquals(BpePreType.LLAMA_SPM, BpePreType.fromPre("llama-spm"));
        assertEquals(BpePreType.DEFAULT, BpePreType.fromPre("default"));
        assertEquals(BpePreType.KIMI_K2, BpePreType.fromPre("kimi-k2"));
        assertEquals(BpePreType.AFMOE, BpePreType.fromPre("afmoe"));
        assertEquals(BpePreType.SEED_CODER, BpePreType.fromPre("seed-coder"));
        assertEquals(BpePreType.GROK2, BpePreType.fromPre("grok-2"));
        assertEquals(BpePreType.TINY_AYA, BpePreType.fromPre("tiny_aya"));
        assertEquals(BpePreType.TEKKEN, BpePreType.fromPre("tekken"));
        assertEquals(BpePreType.WHITESPACE, BpePreType.fromPre("whitespace"));
        assertEquals(BpePreType.GPT2, BpePreType.fromPre("jina-v1-en"));
    }

    @Test
    void llama3DefaultsIgnoreMerges() {
        assertEquals(true, BpePreType.LLAMA3.defaultIgnoreMerges());
        assertEquals(false, BpePreType.QWEN2.defaultIgnoreMerges());
    }

    @Test
    void gemma4UsesRawUtf8() {
        assertEquals(false, BpePreType.GEMMA4.byteEncode());
        assertEquals(true, BpePreType.GPT2.byteEncode());
    }

    @Test
    void unknownPreThrows() {
        assertThrows(IllegalArgumentException.class, () -> BpePreType.fromPre("not-a-real-pre"));
    }
}
