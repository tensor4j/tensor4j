/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.unicode;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.BpePreType;
import java.util.List;
import org.junit.jupiter.api.Test;

class UnicodeRegexSplitTest {

    @Test
    void gemma4NewlinesMatchLlamaPattern() {
        List<String> parts = UnicodeRegexSplit.split("a\nb", BpePreType.GEMMA4.regexes());
        assertEquals(3, parts.size());
        assertEquals("a", parts.get(0));
        assertEquals("\n", parts.get(1));
        assertEquals("b", parts.get(2));
    }

    @Test
    void llama3GroupsDigits() {
        List<String> parts = UnicodeRegexSplit.split("12345", BpePreType.LLAMA3.regexes());
        assertTrue(parts.size() >= 2);
    }

    @Test
    void qwen2SingleDigitSplits() {
        List<String> parts = UnicodeRegexSplit.split("12345", BpePreType.QWEN2.regexes());
        assertEquals(5, parts.size());
    }

    @Test
    void afmoeDigitGrouping() {
        List<String> parts = UnicodeRegexSplit.split("1234567", new String[] {UnicodeRegexPatterns.AFMOE_DIGITS});
        assertEquals(3, parts.size());
        assertEquals("1", parts.get(0));
        assertEquals("234", parts.get(1));
        assertEquals("567", parts.get(2));
    }

    @Test
    void kimiK2GroupsHanCharacters() {
        List<String> parts = UnicodeRegexSplit.split("你好world", BpePreType.KIMI_K2.regexes());
        assertEquals(2, parts.size());
        assertEquals("你好", parts.get(0));
        assertEquals("world", parts.get(1));
    }

    @Test
    void utf8RoundTripThroughCodepoints() {
        int[] cpts = UnicodeCpt.codepointsFromUtf8("café");
        assertEquals("café", UnicodeCpt.sliceToUtf8(cpts, 0, cpts.length));
    }
}
