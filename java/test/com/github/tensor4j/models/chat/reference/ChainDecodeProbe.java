/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.models.chat.reference;

import com.github.tensor4j.models.chat.ChatModel;
import com.github.tensor4j.models.chat.ChatSampler;
import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;

/** Debug chain routing after first decode step. */
public final class ChainDecodeProbe {

    private ChainDecodeProbe() {
    }

    public static void main(String[] args) {
        ChatModel chain = ChatModel.fromGguf(MiniChatGgufBuilder.buildChatDemoModel());
        chain.resetCache();
        int a = ChatSampler.argmax(chain.forward(new int[] {40}));
        int b = ChatSampler.argmax(chain.forward(new int[] {41}));
        int c = ChatSampler.argmax(chain.forward(new int[] {42}));
        System.out.println("after 40 -> " + a);
        System.out.println("after 41 -> " + b);
        System.out.println("after 42 -> " + c);

        chain.resetCache();
        int batch = ChatSampler.argmax(chain.forward(new int[] {40, 41}));
        System.out.println("batch 40,41 last -> " + batch);
    }
}
