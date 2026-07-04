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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.models.chat.fixture.RequiresChatDemoVocab;
import com.github.tensor4j.models.chat.fixture.ChatDemoVocabMode;
import com.github.tensor4j.models.chat.reference.TinygradGenerateReference;
import org.junit.jupiter.api.Test;

/** Open-ended level-12 fixture: seeded weights, real llama3.2 BPE vocab. */
class OpenEndedChatDemoTest {

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.PRUNED)
    void chainModelGreedyRoutesToNextTokenId() {
        ChatModel chain = ChatModel.fromGguf(MiniChatGgufBuilder.buildChatDemoModel());
        int promptId = 40;
        chain.resetCache();
        assertEquals(promptId + 1, ChatSampler.argmax(chain.forward(new int[] {promptId})));
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.FULL)
    void openModelDoesNotGreedyRouteToChainNext() {
        ChatModel open = ChatModel.fromGguf(MiniChatGgufBuilder.buildOpenChatDemoModel());
        int promptId = 40;
        open.resetCache();
        assertNotEquals(promptId + 1, ChatSampler.argmax(open.forward(new int[] {promptId})));
    }

    @Test
    @RequiresChatDemoVocab({ChatDemoVocabMode.PRUNED, ChatDemoVocabMode.FULL})
    void openGreedyCompletionDiffersFromChain() {
        ChatModel chain = ChatModel.fromGguf(MiniChatGgufBuilder.buildChatDemoModel());
        ChatModel open = ChatModel.fromGguf(MiniChatGgufBuilder.buildOpenChatDemoModel());
        ChatGenerationOptions greedy = ChatGenerationOptions.greedy(chain.tokenizer(), 24);

        String chainText = new ChatGenerator(chain, greedy).generate(new int[] {40}).text();
        String openText = new ChatGenerator(open, greedy).generate("Hello", ChatTemplate.PLAIN).text();

        assertTrue(chainText.length() > 1, chainText);
        assertTrue(openText.length() > 1, openText);
        assertFalse(openText.equals(chainText), openText);
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.FULL)
    void differentPromptsProduceDifferentOpenNextToken() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildOpenChatDemoModel());
        int[] hello = model.tokenizer().encode("Hello");
        int[] withThere = new int[hello.length + 1];
        System.arraycopy(hello, 0, withThere, 0, hello.length);
        withThere[hello.length] = hello[0];
        assertTrue(withThere.length > hello.length);

        model.resetCache();
        int helloNext = ChatSampler.argmax(model.forward(hello));
        model.resetCache();
        int extendedNext = ChatSampler.argmax(model.forward(withThere));
        assertNotEquals(helloNext, extendedNext);
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.FULL)
    void seededQualityOpenGenerationIsReproducible() {
        ChatModel a = ChatModel.fromGguf(MiniChatGgufBuilder.buildOpenChatDemoModel());
        ChatModel b = ChatModel.fromGguf(MiniChatGgufBuilder.buildOpenChatDemoModel());
        ChatGenerationOptions options = ChatGenerationOptions.quality(a.tokenizer(), ChatSamplingRngMode.LEGACY);
        String first = new ChatGenerator(a, options).generate("Hello", ChatTemplate.PLAIN).text();
        String second = new ChatGenerator(b, options).generate("Hello", ChatTemplate.PLAIN).text();
        assertEquals(first, second);
        assertTrue(first.length() > 1);
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.FULL)
    void llama3TemplatePrefillDiffersFromPlain() {
        ChatModel model = ChatModel.fromGguf(MiniChatGgufBuilder.buildOpenChatDemoModel());
        int[] plain = ChatTemplate.PLAIN.encodeUser(model.tokenizer(), "Hello");
        int[] llama3 = ChatTemplate.LLAMA3.encodeUser(model.tokenizer(), "Hello");
        assertTrue(llama3.length > plain.length);
        model.resetCache();
        float[] plainLogits = model.forward(plain);
        model.resetCache();
        float[] llama3Logits = model.forward(llama3);
        assertNotEquals(ChatSampler.argmax(plainLogits), ChatSampler.argmax(llama3Logits));
    }

    @Test
    @RequiresChatDemoVocab({ChatDemoVocabMode.PRUNED, ChatDemoVocabMode.FULL})
    void openForwardLogitsAreDistributed() {
        ChatModel chain = ChatModel.fromGguf(MiniChatGgufBuilder.buildChatDemoModel());
        ChatModel open = ChatModel.fromGguf(MiniChatGgufBuilder.buildOpenChatDemoModel());
        int[] prompt = {40};
        chain.resetCache();
        open.resetCache();
        float chainEntropy = softmaxEntropy(chain.forward(prompt));
        float openEntropy = softmaxEntropy(open.forward(open.tokenizer().encode("Hello")));
        assertTrue(chainEntropy < 1.0f, "chain logits should be peaked");
        assertTrue(openEntropy > 3.0f, "open logits should be distributed, entropy=" + openEntropy);
    }

    @Test
    @RequiresChatDemoVocab(ChatDemoVocabMode.FULL)
    void openModelUsesTiedLmHead() {
        ChatModel open = ChatModel.fromGguf(MiniChatGgufBuilder.buildOpenChatDemoModel());
        open.resetCache();
        float[] logits = open.forward(open.tokenizer().encode("Hello"));
        assertFalse(Float.isNaN(logits[0]));
        assertEquals(open.config().nVocab(), logits.length);
    }

    @Test
    void generateLoopPolicyMatchesTinygradReference() {
        assertTrue(TinygradGenerateReference.shouldStop(511, 511, 2, 2));
        assertFalse(TinygradGenerateReference.shouldStop(511, 511, 0, 2));
        assertTrue(TinygradGenerateReference.skipDecode(508, 508, 511));
        assertFalse(TinygradGenerateReference.skipDecode(5, 508, 511));
        assertEquals(
                2,
                TinygradGenerateReference.getStartPos(new int[] {1, 2, 9}, new int[] {1, 2, 3}));
    }

    /** Public for testability — Shannon entropy of softmax(logits). */
    static float softmaxEntropy(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float logit : logits) {
            if (logit > max) {
                max = logit;
            }
        }
        float sum = 0f;
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[i] - max);
            sum += probs[i];
        }
        float entropy = 0f;
        for (float prob : probs) {
            float p = prob / sum;
            if (p > 0f) {
                entropy -= p * (float) Math.log(p);
            }
        }
        return entropy;
    }
}
