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
import java.util.List;
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

    @Test
    void qwenTurnTwoDeltaOpensUserTurnAfterClosedAssistantTurn() {
        ChatTokenizer tokenizer = ChatTokenizer.fromGguf(MiniChatGgufBuilder.buildQwen2TemplateModel().header());
        LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);
        int imEnd = tokenizer.endTurnId();
        int[] assistantBody = new int[] {1};
        List<ChatMessage> afterTurnOne = List.of(
                new ChatMessage("system", ChatTemplate.QWEN2_DEFAULT_SYSTEM),
                new ChatMessage("user", "hello"),
                new ChatMessage("assistant", "Hi", concat(assistantBody, new int[] {imEnd})));
        int prev = applier.tokenCountAfterAssistantTurn(tokenizer, afterTurnOne);
        List<ChatMessage> turnTwo = List.of(
                afterTurnOne.get(0),
                afterTurnOne.get(1),
                afterTurnOne.get(2),
                new ChatMessage("user", "second question"));
        int[] delta = applier.tokenDeltaSince(tokenizer, turnTwo, true, prev);
        assertEquals(tokenizer.tokenIdForText("<|im_start|>"), delta[0]);
        assertEquals(tokenizer.tokenIdForText("user"), delta[1]);
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
