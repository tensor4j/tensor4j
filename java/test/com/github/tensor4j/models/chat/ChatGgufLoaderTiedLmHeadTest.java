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

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertFalse;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.runtime.gguf.GgufFile;
import com.github.tensor4j.support.TensorAssert;
import org.junit.jupiter.api.Test;

/** tinygrad {@code from_gguf} tied {@code output.weight} → {@code token_embd.weight}. */
class ChatGgufLoaderTiedLmHeadTest {

    @Test
    void tiedLmHeadUsesTokenEmbdSlice() {
        GgufFile tied = MiniChatGgufBuilder.buildTiedLmHeadModel();
        assertNull(tied.header().findTensor("output.weight"));
        ChatConfig config = ChatConfig.fromGguf(tied.header());
        ChatWeights weights = ChatGgufLoader.loadViews(tied, config);

        TensorAssert.assertAllClose(
                weights.tokenEmbd().view().dequantizeToFloat(),
                weights.lmHead().view().dequantizeToFloat(),
                1e-6f);

        float[] logits = ChatModel.fromGguf(tied).forward(new int[] {1});
        assertFalse(Float.isNaN(logits[0]));
    }
}
