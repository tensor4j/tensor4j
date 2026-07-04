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

import com.github.tensor4j.runtime.graph.LlamaBlockForward;
import com.github.tensor4j.runtime.infer.InferWeight;

/** Loaded chat model weights (llama.cpp llama_model). */
public final class ChatWeights {

    private final InferWeight tokenEmbd;
    private final InferWeight outputNorm;
    private final InferWeight lmHead;
    private final LlamaBlockForward.Weights[] layers;

    public ChatWeights(
            InferWeight tokenEmbd,
            InferWeight outputNorm,
            InferWeight lmHead,
            LlamaBlockForward.Weights[] layers) {
        this.tokenEmbd = tokenEmbd;
        this.outputNorm = outputNorm;
        this.lmHead = lmHead;
        this.layers = layers.clone();
    }

    public InferWeight tokenEmbd() {
        return tokenEmbd;
    }

    public InferWeight outputNorm() {
        return outputNorm;
    }

    public InferWeight lmHead() {
        return lmHead;
    }

    public LlamaBlockForward.Weights layer(int index) {
        return layers[index];
    }

    public int nLayer() {
        return layers.length;
    }
}
