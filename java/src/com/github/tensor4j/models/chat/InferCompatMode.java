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

/**
 * Inference/chat compatibility layer vs reference runtimes.
 *
 * <p>{@link #TINYGRAD} matches tinygrad {@code apps/llm.py} for parity tests: legacy history
 * default, assistant text re-encode, no early-stop token masking during {@code min_new_tokens}.
 *
 * <p>{@link #LLAMA_CPP} enables llama.cpp {@code simple-chat} behaviors (default for interactive
 * chat): delta KV tokenization, sampled assistant token ids, EOT/BOS guards during min tokens.
 *
 * <p>Weight layout ({@code PERMUTE_QK} vs {@code REVERSE_GGUF_DIMS}) is architecture-driven via
 * {@link ChatConfig#isQwen2Family()} and {@link com.github.tensor4j.runtime.gguf.GgufWeightLayoutMode}.
 *
 * <p>Env: {@code TENSOR4J_INFER_COMPAT=tinygrad|llama} (default {@code llama}).
 */
public enum InferCompatMode {

    TINYGRAD,
    LLAMA_CPP;

    /** {@code TENSOR4J_INFER_COMPAT} — {@code llama} (default) or {@code tinygrad}. */
    public static InferCompatMode fromEnvironment() {
        String raw = System.getenv("TENSOR4J_INFER_COMPAT");
        if (raw == null || raw.isBlank()) {
            return LLAMA_CPP;
        }
        return parseName(raw);
    }

    public static InferCompatMode parseName(String raw) {
        String normalized = raw.trim().toLowerCase(Locale.ROOT);
        if ("tinygrad".equals(normalized) || "tensor4j".equals(normalized) || "legacy".equals(normalized)) {
            return TINYGRAD;
        }
        if ("llama".equals(normalized)
                || "llama.cpp".equals(normalized)
                || "llamacpp".equals(normalized)
                || "llama_cpp".equals(normalized)) {
            return LLAMA_CPP;
        }
        return LLAMA_CPP;
    }

    /** Default {@link ChatHistoryMode} when {@code TENSOR4J_CHAT_HISTORY_MODE} is unset. */
    public ChatHistoryMode defaultHistoryMode() {
        return this == TINYGRAD ? ChatHistoryMode.LEGACY : ChatHistoryMode.LLAMA;
    }

    /** llama.cpp fix: store forwarded ids so delta tokenization matches KV (not tinygrad re-encode). */
    public boolean useSampledAssistantTokenIds() {
        return this == LLAMA_CPP;
    }

    /** llama.cpp fix: mask BOS/EOS/EOT during {@code min_new_tokens} (tinygrad does not). */
    public boolean maskEndTokensDuringMinNewTokens() {
        return this == LLAMA_CPP;
    }
}
