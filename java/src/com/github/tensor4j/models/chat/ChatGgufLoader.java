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
import com.github.tensor4j.runtime.gguf.LlamaQkLayout;
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
                    lazyQueryMatrix(source, prefix + "attn_q.weight", config),
                    lazyKeyMatrix(source, prefix + "attn_k.weight", config),
                    lazyOutMajorMatrix(source, prefix + "attn_v.weight"),
                    lazyOutputMatrix(source, prefix + "attn_output.weight", config),
                    lazyVector(source, prefix + "ffn_norm.weight"),
                    lazyOutMajorMatrix(source, prefix + "ffn_gate.weight"),
                    lazyOutMajorMatrix(source, prefix + "ffn_up.weight"),
                    lazyOutMajorMatrix(source, prefix + "ffn_down.weight"),
                    optionalVector(source, prefix + "attn_q.bias"),
                    optionalVector(source, prefix + "attn_k.bias"),
                    optionalVector(source, prefix + "attn_v.bias"));
        }
        return new ChatWeights(
                lazyEmbedding(source, "token_embd.weight"),
                lazyVector(source, "output_norm.weight"),
                lazyLmHead(source),
                layers);
    }

    /** tinygrad {@code output.weight} or tied {@code token_embd.weight}. */
    private static InferWeight lazyLmHead(GgufTensorSource source) {
        String name = source.header().findTensor("output.weight") != null
                ? "output.weight"
                : "token_embd.weight";
        return lazyOutMajorMatrix(source, name);
    }

    private static InferWeight lazyMatrix(GgufTensorSource source, String name) {
        return InferWeight.lazy(GgufWeightLoader.loadView(source, name), InferWeight.Layout.MATRIX);
    }

    private static InferWeight lazyQkMatrix(GgufTensorSource source, String name, int nHeads) {
        return InferWeight.lazy(
                GgufWeightLoader.loadView(source, name, LlamaQkLayout.PERMUTE_QK, nHeads),
                InferWeight.Layout.MATRIX);
    }

    private static InferWeight lazyQueryMatrix(GgufTensorSource source, String name, ChatConfig config) {
        if (config.isQwen2Family()) {
            return lazyOutMajorMatrix(source, name);
        }
        return lazyQkMatrix(source, name, config.nHead());
    }

    private static InferWeight lazyKeyMatrix(GgufTensorSource source, String name, ChatConfig config) {
        if (config.isQwen2Family()) {
            return lazyOutMajorMatrix(source, name);
        }
        return lazyKMatrix(source, name, config.nHeadKv());
    }

    private static InferWeight lazyKMatrix(GgufTensorSource source, String name, int nKvHeads) {
        return InferWeight.lazy(
                GgufWeightLoader.loadView(source, name, LlamaQkLayout.PERMUTE_QK_KV, nKvHeads),
                InferWeight.Layout.MATRIX);
    }

    private static InferWeight lazyOutMajorMatrix(GgufTensorSource source, String name) {
        return InferWeight.lazy(
                GgufWeightLoader.loadView(source, name, LlamaQkLayout.REVERSE_GGUF_DIMS, 0),
                InferWeight.Layout.MATRIX);
    }

    /** Qwen2 uses output-major {@code attn_output}; llama keeps ggml-native layout. */
    private static InferWeight lazyOutputMatrix(GgufTensorSource source, String name, ChatConfig config) {
        if (config.isQwen2Family()) {
            return lazyOutMajorMatrix(source, name);
        }
        return lazyMatrix(source, name);
    }

    private static InferWeight lazyEmbedding(GgufTensorSource source, String name) {
        return InferWeight.lazy(GgufWeightLoader.loadView(source, name), InferWeight.Layout.EMBEDDING);
    }

    private static InferWeight lazyVector(GgufTensorSource source, String name) {
        return InferWeight.lazy(GgufWeightLoader.loadView(source, name), InferWeight.Layout.VECTOR);
    }

    private static InferWeight optionalVector(GgufTensorSource source, String name) {
        if (source.header().findTensor(name) == null) {
            return null;
        }
        return lazyVector(source, name);
    }
}
