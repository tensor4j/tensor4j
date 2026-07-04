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

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.tensor4j.runtime.infer.InferTensor;
import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.support.TensorAssert;
import org.junit.jupiter.api.Test;

class GgufWeightLoaderTest {

    @Test
    void loadGqaMatricesFromFixture() {
        GgufFile file = MiniChatGgufBuilder.buildIdentityModel();
        InferTensor wq = GgufWeightLoader.loadMatrix(file, "blk.0.attn_q.weight");
        InferTensor wk = GgufWeightLoader.loadMatrix(file, "blk.0.attn_k.weight");
        assertEquals(MiniChatGgufBuilder.N_EMBD, wq.rows());
        assertEquals(MiniChatGgufBuilder.N_EMBD, wq.cols());
        assertEquals(MiniChatGgufBuilder.N_EMBD, wk.rows());
        assertEquals(MiniChatGgufBuilder.N_HEAD_KV * MiniChatGgufBuilder.N_EMBD / MiniChatGgufBuilder.N_HEAD,
                wk.cols());
    }

    @Test
    void loadEmbeddingTransposed() {
        GgufFile file = MiniChatGgufBuilder.buildIdentityModel();
        InferTensor table = GgufWeightLoader.loadEmbedding(file, "token_embd.weight");
        assertEquals(MiniChatGgufBuilder.N_VOCAB, table.rows());
        assertEquals(MiniChatGgufBuilder.N_EMBD, table.cols());
        assertEquals(0.21f, table.get(1, 1), 1e-5f);
    }

    @Test
    void loadMatchesDequantReference() {
        GgufFile file = MiniChatGgufBuilder.buildIdentityModel();
        InferTensor norm = GgufWeightLoader.loadVector(file, "output_norm.weight");
        float[] raw = GgufWeightLoader.dequantize(file, file.header().findTensor("output_norm.weight"));
        TensorAssert.assertAllClose(raw, norm.data(), 1e-6f);
    }
}
