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

import java.util.List;
import org.junit.jupiter.api.Test;

class BpeRegexSplitTest {

    @Test
    void gemma4SplitsOnNewlines() {
        List<String> parts = BpeRegexSplit.split("a\nb", BpePreType.GEMMA4);
        assertEquals(3, parts.size());
        assertEquals("a", parts.get(0));
        assertEquals("\n", parts.get(1));
        assertEquals("b", parts.get(2));
    }

    @Test
    void llama3SplitsDigitsIntoUpToThree() {
        List<String> parts = BpeRegexSplit.split("12345", BpePreType.LLAMA3);
        assertTrue(parts.size() >= 2);
    }

    @Test
    void qwen2SplitsSingleDigits() {
        List<String> parts = BpeRegexSplit.split("12345", BpePreType.QWEN2);
        assertEquals(5, parts.size());
        assertEquals("1", parts.get(0));
        assertEquals("5", parts.get(4));
    }

    @Test
    void llama3GroupsDigitsMoreThanQwen2() {
        List<String> llama3 = BpeRegexSplit.split("12345", BpePreType.LLAMA3);
        List<String> qwen2 = BpeRegexSplit.split("12345", BpePreType.QWEN2);
        assertTrue(llama3.size() < qwen2.size());
    }

    @Test
    void llamaSpmKeepsWholeInput() {
        List<String> parts = BpeRegexSplit.split("ab", BpePreType.LLAMA_SPM);
        assertEquals(1, parts.size());
        assertEquals("ab", parts.get(0));
    }

    @Test
    void defaultAppliesMultiplePatterns() {
        List<String> parts = BpeRegexSplit.split("Hi!!", BpePreType.DEFAULT);
        assertTrue(parts.size() >= 2);
    }
}
