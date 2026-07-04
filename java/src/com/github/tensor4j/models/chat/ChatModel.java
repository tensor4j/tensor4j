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

import com.github.tensor4j.runtime.graph.LlamaBlockForward;

import com.github.tensor4j.runtime.infer.GgmlOps;

import com.github.tensor4j.runtime.infer.InferTensor;

import com.github.tensor4j.runtime.memory.KvCacheStore;

import com.github.tensor4j.runtime.memory.RingKvCache;



/** Minimal llama-style forward: embed → blocks → norm → lm_head (llama.cpp decode). */

public final class ChatModel {



    private final ChatConfig config;

    private final ChatTokenizer tokenizer;

    private final ChatWeights weights;

    private final KvCacheStore[] caches;



    public ChatModel(ChatConfig config, ChatTokenizer tokenizer, ChatWeights weights, KvCacheStore[] caches) {

        if (caches.length != config.nLayer()) {

            throw new IllegalArgumentException("expected " + config.nLayer() + " kv caches");

        }

        this.config = config;

        this.tokenizer = tokenizer;

        this.weights = weights;

        this.caches = caches.clone();

    }



    public static ChatModel fromGguf(GgufTensorSource source) {

        ChatConfig config = ChatConfig.fromGguf(source.header()).withMaxCtx(resolveMaxCtx());

        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(source.header());

        ChatWeights weightViews = ChatGgufLoader.loadViews(source, config);

        return withRingCache(config, tokenizer, weightViews);

    }

    private static int resolveMaxCtx() {
        String env = System.getenv("TENSOR4J_MAX_CTX");
        if (env != null && !env.isBlank()) {
            return Integer.parseInt(env.trim());
        }
        return 2048;
    }

    public static ChatModel withRingCache(ChatConfig config, ChatTokenizer tokenizer, ChatWeights weights) {

        KvCacheStore[] caches = new KvCacheStore[config.nLayer()];

        int headDim = config.headDim();

        for (int i = 0; i < caches.length; i++) {

            caches[i] = new RingKvCache(config.nCtx(), config.nHeadKv(), headDim);

        }

        return new ChatModel(config, tokenizer, weights, caches);

    }



    public int kvLength() {

        return caches[0].nKv();

    }



    public ChatConfig config() {

        return config;

    }



    public ChatTokenizer tokenizer() {

        return tokenizer;

    }

    public ChatWeights weights() {
        return weights;
    }

    /** Live KV caches (one per layer); do not mutate during forward. */
    public KvCacheStore[] caches() {
        return caches;
    }

    public void resetCache() {

        for (KvCacheStore cache : caches) {

            cache.clear();

        }

    }



    /** Prefill/decode tokens; returns logits for the last token [n_vocab]. */

    public float[] forward(int[] tokens) {

        if (tokens.length == 0) {

            throw new IllegalArgumentException("empty token sequence");

        }

        InferTensor x = embed(tokens);

        int past = caches[0].nKv();

        int[] positions = new int[tokens.length];

        for (int i = 0; i < tokens.length; i++) {

            positions[i] = past + i;

        }

        for (int layer = 0; layer < config.nLayer(); layer++) {

            x = LlamaBlockForward.forward(

                    x,

                    weights.layer(layer),

                    caches[layer],

                    config.nHead(),

                    config.nHeadKv(),

                    config.rmsEps(),

                    positions,

                    config.toRopeConfig());

        }

        InferTensor last = lastRow(x);

        InferTensor normed = GgmlOps.rmsNorm(last, weights.outputNorm().tensor(), config.rmsEps());

        InferTensor logits2d = GgmlOps.mulMatOut(normed, weights.lmHead().tensor());

        return logits2d.data();

    }



    public float[] forwardText(String text) {

        return forward(tokenizer.encode(text));

    }



    /** Greedy next-token id from logits. */

    public int sample(int[] tokens) {

        return ChatSampler.argmax(forward(tokens));

    }



    public String sampleText(String text) {

        int next = sample(tokenizer.encode(text));

        return tokenizer.decode(new int[] {next});

    }



    public InferTensor embed(int[] tokens) {

        InferTensor table = weights.tokenEmbd().tensor();

        int nEmbd = config.nEmbd();

        float[] out = new float[tokens.length * nEmbd];

        float[] embd = table.data();

        for (int t = 0; t < tokens.length; t++) {

            int token = tokens[t];

            if (token < 0 || token >= table.rows()) {

                throw new IllegalArgumentException("token id out of range " + token);

            }

            System.arraycopy(embd, token * nEmbd, out, t * nEmbd, nEmbd);

        }

        return InferTensor.of(out, tokens.length, nEmbd);

    }



    private static InferTensor lastRow(InferTensor x) {

        int nEmbd = x.cols();

        float[] row = new float[nEmbd];

        System.arraycopy(x.data(), (x.rows() - 1) * nEmbd, row, 0, nEmbd);

        return InferTensor.vector(row);

    }

}

