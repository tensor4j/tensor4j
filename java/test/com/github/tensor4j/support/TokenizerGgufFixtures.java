/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.support;

import com.github.tensor4j.runtime.gguf.GgufArrayValue;
import com.github.tensor4j.runtime.gguf.GgufConstants;
import com.github.tensor4j.runtime.gguf.GgufHeader;
import com.github.tensor4j.runtime.gguf.GgufKvEntry;
import com.github.tensor4j.runtime.gguf.GgufType;
import java.util.ArrayList;
import java.util.List;

/** Minimal GGUF headers for tokenizer-only tests. */
public final class TokenizerGgufFixtures {

    private TokenizerGgufFixtures() {
    }

    public static GgufHeader header(String pre, String[] tokens) {
        return header(pre, tokens, new String[0], null);
    }

    public static GgufHeader header(String pre, String[] tokens, String[] merges, Boolean ignoreMerges) {
        List<GgufKvEntry> kv = new ArrayList<>();
        kv.add(new GgufKvEntry("tokenizer.ggml.model", GgufType.STRING, "llama"));
        kv.add(new GgufKvEntry("tokenizer.ggml.pre", GgufType.STRING, pre));
        kv.add(new GgufKvEntry("tokenizer.ggml.tokens", GgufType.ARRAY,
                new GgufArrayValue(GgufType.STRING, tokens)));
        kv.add(new GgufKvEntry("tokenizer.ggml.merges", GgufType.ARRAY,
                new GgufArrayValue(GgufType.STRING, merges)));
        kv.add(new GgufKvEntry("tokenizer.ggml.bos_token_id", GgufType.UINT32, 0));
        kv.add(new GgufKvEntry("tokenizer.ggml.eos_token_id", GgufType.UINT32, tokens.length - 1));
        if (ignoreMerges != null) {
            kv.add(new GgufKvEntry("tokenizer.ggml.ignore_merges", GgufType.BOOL, ignoreMerges));
        }
        return new GgufHeader(GgufConstants.VERSION, GgufConstants.DEFAULT_ALIGNMENT, 0, kv, List.of());
    }
}
