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

/** One chat turn for llama.cpp-style template rendering ({@code llama_chat_message}). */
public record ChatMessage(String role, String content, int[] generatedTokenIds) {

    public ChatMessage(String role, String content) {
        this(role, content, null);
    }

    public ChatMessage {
        if (role == null || role.isBlank()) {
            throw new IllegalArgumentException("role required");
        }
        if (content == null) {
            content = "";
        }
        if (generatedTokenIds != null) {
            generatedTokenIds = generatedTokenIds.clone();
        }
    }

    /** Defensive copy for callers. */
    public int[] generatedTokenIds() {
        return generatedTokenIds == null ? null : generatedTokenIds.clone();
    }
}
