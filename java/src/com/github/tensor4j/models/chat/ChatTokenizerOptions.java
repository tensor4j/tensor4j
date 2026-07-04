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

/** Tokenizer options for chat models. */
public record ChatTokenizerOptions(BpeSplitMode splitMode) {

    public static ChatTokenizerOptions defaults() {
        return new ChatTokenizerOptions(BpeSplitMode.LLAMA_UNICODE);
    }

    public static ChatTokenizerOptions javaRegex() {
        return new ChatTokenizerOptions(BpeSplitMode.JAVA_REGEX);
    }

    public static ChatTokenizerOptions llamaUnicode() {
        return new ChatTokenizerOptions(BpeSplitMode.LLAMA_UNICODE);
    }

    public static ChatTokenizerOptions fromSystemProperty() {
        return new ChatTokenizerOptions(BpeSplitMode.fromSystemProperty());
    }
}
