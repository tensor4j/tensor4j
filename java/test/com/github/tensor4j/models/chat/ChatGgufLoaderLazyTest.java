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



import com.github.tensor4j.runtime.gguf.GgufFile;

import com.github.tensor4j.runtime.gguf.GgufHeader;

import com.github.tensor4j.runtime.gguf.GgufTensorSlice;

import com.github.tensor4j.runtime.gguf.GgufTensorSource;

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;

import org.junit.jupiter.api.Test;



class ChatGgufLoaderLazyTest {



    @Test

    void loadViewsStayLazyUntilTensorAccess() {

        GgufFile file = MiniChatGgufBuilder.buildIdentityModel();

        TrackingSource source = new TrackingSource(file);

        ChatConfig config = ChatConfig.fromGguf(source.header());

        ChatWeights weights = ChatGgufLoader.loadViews(source, config);



        assertTrue(weights.tokenEmbd().isLazy());

        assertTrue(weights.layer(0).wq().isLazy());

        assertEquals(0, source.tensorBytesCalls);



        weights.tokenEmbd().tensor();

        assertEquals(0, source.tensorBytesCalls);

    }



    private static final class TrackingSource implements GgufTensorSource {



        private final GgufFile delegate;

        private int tensorBytesCalls;



        private TrackingSource(GgufFile delegate) {

            this.delegate = delegate;

        }



        @Override

        public GgufHeader header() {

            return delegate.header();

        }



        @Override

        public byte[] tensorBytes(String name) {

            tensorBytesCalls++;

            return delegate.tensorBytes(name);

        }



        @Override

        public GgufTensorSlice tensorSlice(String name) {

            return delegate.tensorSlice(name);

        }

    }

}

