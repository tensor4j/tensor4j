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

/** BPE pre-split backend ({@code unicode_regex_split} vs Java regex). */
public enum BpeSplitMode {

    /** llama.cpp codepoint + collapsed-unicode splitters ({@code runtime.unicode}) — default. */
    LLAMA_UNICODE,

    /** Fast Java {@link java.util.regex.Pattern} splitter (opt-in). */
    JAVA_REGEX;

    public static BpeSplitMode fromSystemProperty() {
        String value = System.getProperty("tensor4j.bpe.split", "llama_unicode");
        if ("java_regex".equalsIgnoreCase(value) || "java".equalsIgnoreCase(value)) {
            return JAVA_REGEX;
        }
        return LLAMA_UNICODE;
    }
}
