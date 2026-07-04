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

/** Entropy source for Gumbel / multinomial draws during chat sampling. */
public enum ChatSamplingRngMode {

    /** {@link java.security.SecureRandom} — default for interactive / production sampling. */
    SECURE,

    /** Seeded {@link java.util.Random} — reproducible CI and parity tests ({@code TENSOR4J_CHAT_SEED}). */
    LEGACY;

    public static ChatSamplingRngMode fromString(String value) {
        if (value == null || value.isBlank()) {
            return SECURE;
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        if ("legacy".equals(normalized) || "random".equals(normalized) || "seeded".equals(normalized)) {
            return LEGACY;
        }
        if ("secure".equals(normalized) || "cryptographic".equals(normalized)) {
            return SECURE;
        }
        return SECURE;
    }
}
