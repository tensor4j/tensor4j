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

import java.util.Locale;

/** Completion policy — quality sampling vs fast greedy argmax. */
public enum ChatGenerationMode {

    /** Temperature + top-p, suppress early EOS (default for demos). */
    QUALITY,

    /** Argmax only — deterministic smoke tests. */
    GREEDY;

    public static ChatGenerationMode fromString(String value) {
        if (value == null || value.isBlank()) {
            return QUALITY;
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        if ("greedy".equals(normalized) || "fast".equals(normalized) || "argmax".equals(normalized)) {
            return GREEDY;
        }
        if ("quality".equals(normalized) || "sample".equals(normalized) || "chat".equals(normalized)) {
            return QUALITY;
        }
        return QUALITY;
    }
}
