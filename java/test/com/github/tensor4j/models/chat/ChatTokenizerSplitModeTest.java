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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.tensor4j.support.TokenizerGgufFixtures;
import org.junit.jupiter.api.Test;

class ChatTokenizerSplitModeTest {

    @Test
    void asciiFixtureMatchesAcrossSplitModes() {
        String[] tokens = {"<s>", "a", "b", "</s>"};
        ChatTokenizer javaMode = ChatTokenizer.fromGguf(
                TokenizerGgufFixtures.header("llama-spm", tokens),
                ChatTokenizerOptions.javaRegex());
        ChatTokenizer llamaMode = ChatTokenizer.fromGguf(
                TokenizerGgufFixtures.header("llama-spm", tokens),
                ChatTokenizerOptions.defaults());
        assertEquals(BpeSplitMode.JAVA_REGEX, javaMode.splitMode());
        assertEquals(BpeSplitMode.LLAMA_UNICODE, llamaMode.splitMode());
        assertArrayEquals(javaMode.encode("ab"), llamaMode.encode("ab"));
    }

    @Test
    void defaultsUseLlamaUnicode() {
        assertEquals(BpeSplitMode.LLAMA_UNICODE, ChatTokenizerOptions.defaults().splitMode());
    }

    @Test
    void llama3WholeWordLookupInUnicodeMode() {
        String[] tokens = {"<s>", "Hello", "!", "</s>"};
        ChatTokenizer tok = ChatTokenizer.fromGguf(
                TokenizerGgufFixtures.header("llama3", tokens),
                ChatTokenizerOptions.defaults());
        assertArrayEquals(new int[] {1}, tok.encode("Hello"));
    }

    @Test
    void systemPropertySelectsJavaRegex() {
        String previous = System.getProperty("tensor4j.bpe.split");
        try {
            System.setProperty("tensor4j.bpe.split", "java_regex");
            assertEquals(BpeSplitMode.JAVA_REGEX, ChatTokenizerOptions.fromSystemProperty().splitMode());
        } finally {
            if (previous == null) {
                System.clearProperty("tensor4j.bpe.split");
            } else {
                System.setProperty("tensor4j.bpe.split", previous);
            }
        }
    }

    @Test
    void unsetSystemPropertyDefaultsToLlamaUnicode() {
        String previous = System.getProperty("tensor4j.bpe.split");
        try {
            System.clearProperty("tensor4j.bpe.split");
            assertEquals(BpeSplitMode.LLAMA_UNICODE, BpeSplitMode.fromSystemProperty());
        } finally {
            if (previous == null) {
                System.clearProperty("tensor4j.bpe.split");
            } else {
                System.setProperty("tensor4j.bpe.split", previous);
            }
        }
    }
}
