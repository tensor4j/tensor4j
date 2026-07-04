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

/** Output of one {@link ChatGenerator} completion pass. */
public record ChatGenerationResult(
        String text,
        int tokenCount,
        String mode,
        int prefixReuseTokens,
        int[] generatedTokenIds,
        int[] forwardedTokenIds) {
}
