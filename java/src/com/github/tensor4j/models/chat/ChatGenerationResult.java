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
        int[] forwardedTokenIds,
        ChatGenerationStopReason stopReason,
        int stopTokenId,
        int tokensRemaining,
        ChatGenerationStep[] steps) {

    /** Backward-compatible constructor without stop metadata. */
    public ChatGenerationResult(
            String text,
            int tokenCount,
            String mode,
            int prefixReuseTokens,
            int[] generatedTokenIds,
            int[] forwardedTokenIds) {
        this(
                text,
                tokenCount,
                mode,
                prefixReuseTokens,
                generatedTokenIds,
                forwardedTokenIds,
                ChatGenerationStopReason.UNKNOWN,
                -1,
                0,
                new ChatGenerationStep[0]);
    }

    /** Backward-compatible constructor without tokensRemaining. */
    public ChatGenerationResult(
            String text,
            int tokenCount,
            String mode,
            int prefixReuseTokens,
            int[] generatedTokenIds,
            int[] forwardedTokenIds,
            ChatGenerationStopReason stopReason,
            int stopTokenId,
            ChatGenerationStep[] steps) {
        this(
                text,
                tokenCount,
                mode,
                prefixReuseTokens,
                generatedTokenIds,
                forwardedTokenIds,
                stopReason,
                stopTokenId,
                0,
                steps);
    }
}
