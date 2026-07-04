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

/** Llama Q/K row layout transform when loading GGUF matrices. */
public enum LlamaQkLayout {
    NONE,
    /** Q rows ({@code n_head * head_dim}) interleaved RoPE permute; square so no dim reverse. */
    PERMUTE_QK,
    /** K: reverse GGUF dims then kv-head row permute → output-major {@code [n_kv*hd, n_embd]}. */
    PERMUTE_QK_KV,
    /** V (and similar): reverse GGUF dims only → output-major {@code [n_out, n_embd]}. */
    REVERSE_GGUF_DIMS
}
