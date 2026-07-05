/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.runtime.gguf;

import java.util.Locale;

/**
 * GGUF matrix layout when loading output-major weights ({@link LlamaQkLayout#REVERSE_GGUF_DIMS}).
 *
 * <p>{@link #TINYGRAD} (default) applies {@link GgufLlamaLayout#reverseGgufDims} — matches tinygrad
 * {@code reshape(*reversed(dims))}.
 *
 * <p>{@link #METADATA_ONLY} is the pre-fix behavior (swap rows/cols without transposing data) kept
 * for debugging only.
 *
 * <p>Env: {@code TENSOR4J_GGUF_WEIGHT_LAYOUT=tinygrad|metadata} (default {@code tinygrad}).
 */
public enum GgufWeightLayoutMode {

    TINYGRAD,
    METADATA_ONLY;

    public static GgufWeightLayoutMode fromEnvironment() {
        String raw = System.getenv("TENSOR4J_GGUF_WEIGHT_LAYOUT");
        if (raw == null || raw.isBlank()) {
            return TINYGRAD;
        }
        return parseName(raw);
    }

    public static GgufWeightLayoutMode parseName(String raw) {
        String normalized = raw.trim().toLowerCase(Locale.ROOT);
        if ("metadata".equals(normalized) || "legacy".equals(normalized) || "wrong".equals(normalized)) {
            return METADATA_ONLY;
        }
        return TINYGRAD;
    }

    public boolean transposeReverseGgufDims() {
        return this == TINYGRAD;
    }
}
