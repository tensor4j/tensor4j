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
import com.github.tensor4j.models.chat.reference.ChatForwardTrace.Checkpoint;
import com.github.tensor4j.runtime.gguf.MmappedGgufFile;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

/** JSON-ish layer trace for {@code tools/compare_forward_layers.py}. */
public final class LayerForwardProbe {

    private static final String PROMPT = "Hello";

    private LayerForwardProbe() {
    }

    public static void main(String[] args) throws Exception {
        String ggufPath = System.getenv("TENSOR4J_GGUF_PATH");
        if (ggufPath == null || ggufPath.isBlank()) {
            throw new IllegalStateException("set TENSOR4J_GGUF_PATH");
        }
        Path path = Paths.get(ggufPath);
        try (MmappedGgufFile mapped = MmappedGgufFile.open(path)) {
            ChatModel model = ChatModel.fromGguf(mapped);
            int[] tokens = model.tokenizer().encode(PROMPT);
            model.resetCache();
            List<Checkpoint> checkpoints = ChatForwardTrace.tracePrefill(model, tokens);
            System.out.print(toJson(tokens, checkpoints));
        }
    }

    private static String toJson(int[] tokens, List<Checkpoint> checkpoints) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"prompt\":\"").append(PROMPT).append("\",\"token_ids\":")
                .append(Arrays.toString(tokens)).append(",\"checkpoints\":[");
        for (int i = 0; i < checkpoints.size(); i++) {
            Checkpoint cp = checkpoints.get(i);
            if (i > 0) {
                sb.append(',');
            }
            sb.append("{\"name\":\"").append(cp.name()).append("\",")
                    .append("\"max_abs\":").append(cp.maxAbs()).append(',')
                    .append("\"l2\":").append(cp.l2()).append(',')
                    .append("\"head\":").append(Arrays.toString(cp.head())).append('}');
        }
        sb.append("]}");
        return sb.toString();
    }
}
