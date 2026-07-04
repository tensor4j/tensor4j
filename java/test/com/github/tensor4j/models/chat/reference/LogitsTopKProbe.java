/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.chat.reference;

import com.github.tensor4j.models.chat.ChatModel;
import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.runtime.gguf.MmappedGgufFile;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * Prints top-10 prefill logits for {@code Hello} — compare with
 * {@code python tools/capture_tinygrad_hello_logits.py}.
 */
public final class LogitsTopKProbe {

    private static final int TOP_K = 10;
    private static final String PROMPT = "Hello";

    private LogitsTopKProbe() {
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== tensor4j LogitsTopKProbe (prompt=\"" + PROMPT + "\") ===");
        probe("mini_open_fixture", ChatModel.fromGguf(MiniChatGgufBuilder.buildOpenChatDemoModel()));

        String ggufPath = System.getenv("TENSOR4J_GGUF_PATH");
        if (ggufPath != null && !ggufPath.isBlank()) {
            Path path = Paths.get(ggufPath);
            try (MmappedGgufFile mapped = MmappedGgufFile.open(path)) {
                probe("external_gguf:" + path.getFileName(), ChatModel.fromGguf(mapped));
            }
        } else {
            System.out.println("(set TENSOR4J_GGUF_PATH for real-weight top-k)");
        }
    }

    private static void probe(String label, ChatModel model) {
        int[] tokens = model.tokenizer().encode(PROMPT);
        model.resetCache();
        float[] logits = model.forward(tokens);
        System.out.println();
        System.out.println("[" + label + "]");
        System.out.println("  tokens=" + Arrays.toString(tokens));
        System.out.println("  vocab=" + model.config().nVocab()
                + "  layers=" + model.config().nLayer()
                + "  kv=" + model.kvLength());
        System.out.println(LogitsTopK.formatReport("  top-" + TOP_K + " logits:", logits, model.tokenizer(), TOP_K));
    }
}
