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

import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder;
import com.github.tensor4j.runtime.gguf.GgufFile;
import org.junit.jupiter.api.Test;

/** Llama 3 multi-turn prompt structure (tinygrad {@code apps/llm.py} interactive loop). */
class ChatTemplateMultiTurnTest {

    @Test
    void turnTwoPromptInsertsEotBeforeNextUserTurn() {
        GgufFile file = MiniChatGgufBuilder.buildLlama3TemplateModel();
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(file.header());
        int eos = tokenizer.eosId();

        int[] turnOneResponse = new int[] {1};
        int[] afterTurnOne = concat(
                ChatTemplate.LLAMA3.encodePromptForGeneration(tokenizer, "hello"),
                turnOneResponse,
                new int[] {eos});

        int[] turnTwoPrompt = concat(
                afterTurnOne,
                ChatTemplate.LLAMA3.encodeUserTurn(tokenizer, "second question"),
                ChatTemplate.LLAMA3.encodeAssistantPrime(tokenizer));

        assertEquals(eos, afterTurnOne[afterTurnOne.length - 1]);
        assertEquals(
                ChatTemplate.LLAMA3.encodeUserTurn(tokenizer, "second question").length,
                turnTwoPrompt.length - afterTurnOne.length - ChatTemplate.LLAMA3.encodeAssistantPrime(tokenizer).length);
    }

    private static int[] concat(int[]... parts) {
        int len = 0;
        for (int[] part : parts) {
            len += part.length;
        }
        int[] out = new int[len];
        int pos = 0;
        for (int[] part : parts) {
            System.arraycopy(part, 0, out, pos, part.length);
            pos += part.length;
        }
        return out;
    }
}
