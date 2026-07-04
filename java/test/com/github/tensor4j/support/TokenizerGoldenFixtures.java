/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.support;

import com.github.tensor4j.models.chat.BpePreType;
import com.github.tensor4j.runtime.unicode.UnicodeRegexPatterns;

/**
 * Golden BPE pre-split vectors aligned with llama.cpp {@code unicode_regex_split_custom}
 * and collapsed-regex fallback ({@code src/unicode.cpp}).
 */
public final class TokenizerGoldenFixtures {

    public record GoldenCase(String name, String[] regexes, String input, String[] expectedParts) {
    }

    /** Canonical llama.cpp parity cases — do not change without re-validating against llama.cpp. */
    public static final GoldenCase[] UNICODE_SPLIT_CASES = {
            golden(BpePreType.LLAMA3, "12345", "123", "45"),
            golden(BpePreType.LLAMA3, "Hello", "Hello"),
            golden(BpePreType.LLAMA3, "Hello world", "Hello", " world"),
            golden(BpePreType.LLAMA3, "a\nb", "a", "\n", "b"),
            golden(BpePreType.QWEN2, "12345", "1", "2", "3", "4", "5"),
            golden(BpePreType.QWEN2, "Hi!", "Hi", "!"),
            golden(BpePreType.KIMI_K2, "你好world", "你好", "world"),
            golden(BpePreType.KIMI_K2, "Hello123", "Hello", "123"),
            custom("afmoe-digits", new String[] {UnicodeRegexPatterns.AFMOE_DIGITS}, "1234567", "1", "234", "567"),
            custom("afmoe-digits", new String[] {UnicodeRegexPatterns.AFMOE_DIGITS}, "12", "12"),
            custom("afmoe-digits", new String[] {UnicodeRegexPatterns.AFMOE_DIGITS}, "1234", "1", "234"),
            custom("tiny-aya-digits", new String[] {UnicodeRegexPatterns.TINY_AYA_DIGITS}, "1234567", "1", "234", "567"),
            golden(BpePreType.GEMMA4, "a\nb", "a", "\n", "b"),
            golden(BpePreType.JAIS2, "Hi  there", "Hi", " ", " there"),
    };

    private static GoldenCase golden(BpePreType preType, String input, String... expectedParts) {
        return new GoldenCase(preType.name(), preType.regexes(), input, expectedParts);
    }

    private static GoldenCase custom(String name, String[] regexes, String input, String... expectedParts) {
        return new GoldenCase(name, regexes, input, expectedParts);
    }

    private TokenizerGoldenFixtures() {
    }
}
