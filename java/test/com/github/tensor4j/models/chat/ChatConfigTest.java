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

import com.github.tensor4j.runtime.infer.RopeScalingType;
import com.github.tensor4j.models.chat.BpePreType;
import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import org.junit.jupiter.api.Test;

class ChatConfigTest {

    @Test
    void parsesPartialRopeDim() {
        ChatConfig config = ChatConfig.fromGguf(MiniChatGgufBuilder.buildIdentityModel().header());
        assertEquals(4, config.ropeDim());
        assertEquals(8, config.headDim());
        assertEquals(4, config.toRopeConfig().ropeDim());
    }

    @Test
    void parsesLlamaSpmPre() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildIdentityModel().header());
        assertEquals(BpePreType.LLAMA_SPM, tokenizer.preType());
    }

    @Test
    void parsesYarnMetadata() {
        ChatConfig config = ChatConfig.fromGguf(MiniChatGgufBuilder.buildYarnModel().header());
        assertEquals(RopeScalingType.YARN, config.ropeScaling());
        assertEquals(2.0f, config.ropeScaleFactor(), 1e-5f);
        assertEquals(4, config.ropeOrigCtx());
    }

    @Test
    void parsesQwen2Architecture() {
        ChatConfig config = ChatConfig.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel().header());
        assertEquals("qwen2", config.architecture());
        assertTrue(config.isQwen2Family());
    }
}
