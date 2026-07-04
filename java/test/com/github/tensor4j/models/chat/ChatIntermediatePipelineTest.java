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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.models.chat.fixture.RequiresChatDemoVocab;
import com.github.tensor4j.models.chat.fixture.ChatDemoVocabMode;
import com.github.tensor4j.models.chat.reference.TinygradGenerateReference;
import com.github.tensor4j.runtime.infer.InferTensor;
import com.github.tensor4j.support.TensorAssert;
import org.junit.jupiter.api.Test;

/**
 * Intermediate chat pipeline tests between unit parity and full completion ITs.
 *
 * <p>Stages: tokenize → embed → batch/incremental forward → greedy decode loop (tinygrad
 * {@code apps/llm.py} {@code Transformer.forward} + {@code generate()} prefill/decode split).
 */
class ChatIntermediatePipelineTest {

    @Test
    void embedRowsMatchTokenEmbeddingTable() {
        var file = MiniChatGgufBuilder.buildIdentityModel();
        ChatConfig config = ChatConfig.fromGguf(file.header());
        InferTensor table = ChatGgufLoader.loadViews(file, config).tokenEmbd().tensor();
        ChatModel model = ChatModel.fromGguf(file);
        int[] tokens = {1, 2};
        InferTensor embedded = model.embed(tokens);
        assertEquals(2, embedded.rows());
        assertEquals(MiniChatGgufBuilder.N_EMBD, embedded.cols());
        for (int row = 0; row < tokens.length; row++) {
            TensorAssert.assertAllClose(rowFromTable(table, tokens[row]), rowFromTensor(embedded, row), 0f);
        }
    }

    @Test
    void batchPrefillMatchesIncrementalDecodeOnIdentityFixture() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        int[] prompt = {0, 1, 2};

        model.resetCache();
        float[] batch = model.forward(prompt);

        model.resetCache();
        float[] incremental = null;
        for (int token : prompt) {
            incremental = model.forward(new int[] {token});
        }

        assertArrayEquals(batch, incremental, 1e-5f);
    }

    @Test
    void chunkedPrefillMatchesSingleForwardOnIdentityFixture() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        int[] prompt = {0, 1, 2, 3};
        model.resetCache();
        float[] single = model.forward(prompt);
        model.resetCache();
        float[] chunked = ChatGenerator.prefillChunked(model, prompt, 0, 2);
        assertArrayEquals(single, chunked, 1e-5f);
    }

    @Test
    void tokenizeForwardArgmaxPipelineOnLlama3BpeFixture() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildLlama3BpeModel());
        int[] hello = model.tokenizer().encode("Hello");
        assertArrayEquals(new int[] {1}, hello);
        float[] logits = model.forward(hello);
        assertEquals(5, logits.length);
        assertEquals(4, ChatSampler.argmax(logits));
    }

    @Test
    void templatePrefillLogitsDifferFromBareMessageForward() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildLlama3TemplateModel());
        int[] plain = model.tokenizer().encode("Hello");
        int[] templated = ChatTemplate.LLAMA3.encodeUser(model.tokenizer(), "Hello");
        assertTrue(templated.length > plain.length);
        model.resetCache();
        float[] plainLogits = model.forward(plain);
        model.resetCache();
        float[] templateLogits = model.forward(templated);
        org.junit.jupiter.api.Assertions.assertFalse(
                java.util.Arrays.equals(plainLogits, templateLogits),
                "llama3 role header must change prefill logits");
    }

    @Test
    void generatePrefillFlagsMatchTinygradChunkedPrefillPattern() {
        boolean[] flags = ChatGenerator.forwardPrefillFlags(9, 4, 3);
        assertEquals(5, flags.length);
        assertTrue(flags[0] && flags[1] && flags[2]);
        assertTrue(!flags[3] && !flags[4]);
    }

    @Test
    void eosStopPolicyBlocksImmediateGreedyCompletionOnIdentityEosLogit() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildIdentityModel());
        int eos = model.tokenizer().eosId();
        int next = ChatSampler.argmax(model.forward(new int[] {1, 2}));
        assertEquals(eos, next);
        assertTrue(TinygradGenerateReference.shouldStop(next, eos, 0, 0));

        ChatGenerator generator = new ChatGenerator(model, ChatGenerationOptions.greedy(model.tokenizer(), 4));
        var result = generator.generate("ab", ChatTemplate.PLAIN);
        assertEquals(0, result.tokenCount());
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void chainSingleHopGreedyRoutesToNextTokenId() {
        ChatModel chain = ChatModel.fromGguf(MiniChatGgufBuilder.buildChatDemoModel());
        int promptId = 40;
        chain.resetCache();
        assertEquals(promptId + 1, ChatSampler.argmax(chain.forward(new int[] {promptId})));
    }

    /**
     * Chain lm_head routes {@code i → i+1} when embeddings pass through blocks unchanged
     * ({@code attn_output}=0, FFN no-op in {@link MiniChatGgufBuilder#buildChatDemoModel()}).
     */
    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void chainGreedyMultiHopAdvancesTokenIds() {
        ChatModel chain = ChatModel.fromGguf(MiniChatGgufBuilder.buildChatDemoModel());
        chain.resetCache();
        chain.forward(new int[] {40});
        assertEquals(42, ChatSampler.argmax(chain.forward(new int[] {41})),
                "incremental decode after prefill on 40 should route 41→42");

        ChatGenerator generator = new ChatGenerator(chain, ChatGenerationOptions.greedy(chain.tokenizer(), 5));
        assertArrayEquals(new int[] {41, 42, 43, 44, 45},
                generator.generate(new int[] {40}).generatedTokenIds());

        chain.resetCache();
        assertEquals(42, ChatSampler.argmax(chain.forward(new int[] {40, 41})),
                "batch prefill last token should also route 41→42");
    }

    private static float[] rowFromTable(InferTensor table, int tokenId) {
        int nEmbd = table.cols();
        float[] row = new float[nEmbd];
        System.arraycopy(table.data(), tokenId * nEmbd, row, 0, nEmbd);
        return row;
    }

    private static float[] rowFromTensor(InferTensor tensor, int row) {
        int nEmbd = tensor.cols();
        float[] out = new float[nEmbd];
        System.arraycopy(tensor.data(), row * nEmbd, out, 0, nEmbd);
        return out;
    }
}
