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

import com.github.tensor4j.runtime.gguf.GgufTensorSource;
import com.github.tensor4j.runtime.gguf.GgufWeightLoader;
import com.github.tensor4j.runtime.graph.LlamaBlockForward;
import com.github.tensor4j.runtime.infer.InferWeight;

/** Load chat weights from GGUF views (llama tensor naming; lazy dequant on first use). */
public final class ChatGgufLoader {

    private ChatGgufLoader() {
    }

    public static ChatWeights load(GgufTensorSource source, ChatConfig config) {
        return loadViews(source, config);
    }

    public static ChatWeights loadViews(GgufTensorSource source, ChatConfig config) {
        LlamaBlockForward.Weights[] layers = new LlamaBlockForward.Weights[config.nLayer()];
        for (int i = 0; i < config.nLayer(); i++) {
            String prefix = "blk." + i + ".";
            layers[i] = new LlamaBlockForward.Weights(
                    lazyVector(source, prefix + "attn_norm.weight"),
                    lazyMatrix(source, prefix + "attn_q.weight"),
                    lazyMatrix(source, prefix + "attn_k.weight"),
                    lazyMatrix(source, prefix + "attn_v.weight"),
                    lazyMatrix(source, prefix + "attn_output.weight"),
                    lazyVector(source, prefix + "ffn_norm.weight"),
                    lazyMatrix(source, prefix + "ffn_gate.weight"),
                    lazyMatrix(source, prefix + "ffn_up.weight"),
                    lazyMatrix(source, prefix + "ffn_down.weight"));
        }
        return new ChatWeights(
                lazyEmbedding(source, "token_embd.weight"),
                lazyVector(source, "output_norm.weight"),
                lazyMatrix(source, "output.weight"),
                layers);
    }

    private static InferWeight lazyMatrix(GgufTensorSource source, String name) {
        return InferWeight.lazy(GgufWeightLoader.loadView(source, name), InferWeight.Layout.MATRIX);
    }

    private static InferWeight lazyEmbedding(GgufTensorSource source, String name) {
        return InferWeight.lazy(GgufWeightLoader.loadView(source, name), InferWeight.Layout.EMBEDDING);
    }

    private static InferWeight lazyVector(GgufTensorSource source, String name) {
        return InferWeight.lazy(GgufWeightLoader.loadView(source, name), InferWeight.Layout.VECTOR);
    }
}
