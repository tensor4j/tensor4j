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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.runtime.gguf.GgufFile;
import com.github.tensor4j.runtime.graph.LlamaBlockForward;
import com.github.tensor4j.runtime.infer.InferTensor;
import com.github.tensor4j.runtime.infer.RopeConfig;
import com.github.tensor4j.runtime.memory.KvCache;
import com.github.tensor4j.support.TensorAssert;
import org.junit.jupiter.api.Test;

/** Full decoder block + chat forward with llama-track lazy mmap dequant on first use. */
class LazyWeightFullGraphTest {

    private static final RopeConfig NO_ROPE = RopeConfig.disabled();

    @Test
    void q4LazyBlockForwardMatchesF32EagerBlock() {
        GgufFile q4File = MiniChatGgufBuilder.buildQ4Model();
        GgufFile f32File = MiniChatGgufBuilder.buildIdentityModel();
        ChatConfig config = ChatConfig.fromGguf(q4File.header());

        ChatWeights q4Weights = ChatGgufLoader.loadViews(q4File, config);
        ChatWeights f32Weights = ChatGgufLoader.loadViews(f32File, config);
        assertTrue(q4Weights.layer(0).wq().isLazy());

        KvCache q4Cache = new KvCache(MiniChatGgufBuilder.N_CTX, MiniChatGgufBuilder.N_HEAD_KV,
                config.headDim());
        KvCache f32Cache = new KvCache(MiniChatGgufBuilder.N_CTX, MiniChatGgufBuilder.N_HEAD_KV,
                config.headDim());

        InferTensor x = smokeInput(MiniChatGgufBuilder.N_EMBD);
        InferTensor q4Out = LlamaBlockForward.forward(
                x, q4Weights.layer(0), q4Cache,
                config.nHead(), config.nHeadKv(), config.rmsEps(), new int[] {0}, NO_ROPE);
        InferTensor f32Out = LlamaBlockForward.forward(
                x, f32Weights.layer(0), f32Cache,
                config.nHead(), config.nHeadKv(), config.rmsEps(), new int[] {0}, NO_ROPE);

        TensorAssert.assertAllClose(f32Out.data(), q4Out.data(), 0.25f);
    }

    @Test
    void q4LazyChatForwardArgmaxMatchesF32() {
        GgufFile q4File = MiniChatGgufBuilder.buildQ4Model();
        GgufFile f32File = MiniChatGgufBuilder.buildIdentityModel();

        ChatModel q4Model = ChatModel.withRingCache(
                ChatConfig.fromGguf(q4File.header()),
                ChatTokenizer.fromGguf(q4File.header()),
                ChatGgufLoader.loadViews(q4File, ChatConfig.fromGguf(q4File.header())));
        ChatModel f32Model = ChatModel.withRingCache(
                ChatConfig.fromGguf(f32File.header()),
                ChatTokenizer.fromGguf(f32File.header()),
                ChatGgufLoader.loadViews(f32File, ChatConfig.fromGguf(f32File.header())));

        float[] q4Logits = q4Model.forward(new int[] {1});
        float[] f32Logits = f32Model.forward(new int[] {1});
        assertEquals(f32Logits.length, q4Logits.length);
        assertEquals(ChatSampler.argmax(f32Logits), ChatSampler.argmax(q4Logits));
    }

    private static InferTensor smokeInput(int nEmbd) {
        float[] data = new float[nEmbd];
        for (int i = 0; i < nEmbd; i++) {
            data[i] = (i + 1) * 0.1f;
        }
        return InferTensor.of(data, 1, nEmbd);
    }
}
