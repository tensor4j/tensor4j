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



import com.github.tensor4j.runtime.gguf.GgufFile;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;

import org.junit.jupiter.api.Test;



class ChatModelTest {



    private static ChatModel modelFrom(GgufFile file) {

        ChatConfig config = ChatConfig.fromGguf(file.header());

        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(file.header());

        ChatWeights weights = ChatGgufLoader.loadViews(file, config);

        return ChatModel.withRingCache(config, tokenizer, weights);

    }



    @Test

    void fullForwardFromGguf() {

        ChatModel model = modelFrom(MiniChatGgufBuilder.buildIdentityModel());

        float[] logits = model.forward(new int[] {1, 2});

        assertEquals(MiniChatGgufBuilder.N_VOCAB, logits.length);

    }



    @Test

    void forwardTextUsesTokenizer() {

        ChatModel model = modelFrom(MiniChatGgufBuilder.buildIdentityModel());

        assertArrayEquals(new int[] {1, 2}, model.tokenizer().encode("ab"));

        float[] viaText = model.forwardText("ab");

        model.resetCache();

        float[] viaIds = model.forward(new int[] {1, 2});

        assertArrayEquals(viaIds, viaText);

    }



    @Test

    void incrementalDecodeUsesRingCache() {

        ChatModel model = modelFrom(MiniChatGgufBuilder.buildIdentityModel());

        float[] first = model.forward(new int[] {0});

        float[] second = model.forward(new int[] {1});

        assertEquals(MiniChatGgufBuilder.N_VOCAB, first.length);

        assertEquals(MiniChatGgufBuilder.N_VOCAB, second.length);

    }



    @Test

    void ringCacheEvictionBoundsMemory() {

        GgufFile file = MiniChatGgufBuilder.buildIdentityModel();

        ChatConfig base = ChatConfig.fromGguf(file.header());

        ChatConfig config = new ChatConfig(

                base.nVocab(),

                base.nEmbd(),

                base.nHead(),

                base.nHeadKv(),

                base.nLayer(),

                2,

                base.rmsEps(),

                base.ropeBase(),

                base.ropeDim(),

                base.ropeScaling(),

                base.ropeScaleFactor(),

                base.ropeOrigCtx(),

                base.yarnExtFactor(),

                base.yarnAttnFactor(),

                base.architecture());

        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(file.header());

        ChatWeights weights = ChatGgufLoader.loadViews(file, config);

        ChatModel model = ChatModel.withRingCache(config, tokenizer, weights);



        model.forward(new int[] {0, 1});

        assertEquals(2, model.kvLength());

        model.forward(new int[] {2});

        assertEquals(2, model.kvLength());

        float[] logits = model.forward(new int[] {3});

        assertEquals(MiniChatGgufBuilder.N_VOCAB, logits.length);

    }



    @Test

    void lazyWeightsMaterializeOnForward() {

        GgufFile file = MiniChatGgufBuilder.buildIdentityModel();

        ChatConfig config = ChatConfig.fromGguf(file.header());

        ChatWeights weights = ChatGgufLoader.loadViews(file, config);

        assertTrue(weights.tokenEmbd().isLazy());

        ChatModel model = ChatModel.withRingCache(config, ChatTokenizer.fromGguf(file.header()), weights);

        model.forward(new int[] {1});

        assertTrue(weights.lmHead().tensor().data().length > 0);

    }

}

