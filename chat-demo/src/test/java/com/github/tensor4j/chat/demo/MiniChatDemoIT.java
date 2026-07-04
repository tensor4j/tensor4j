/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.chat.demo;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.BpePreType;
import com.github.tensor4j.models.chat.ChatGenerationMode;
import com.github.tensor4j.models.chat.ChatGenerationOptions;
import com.github.tensor4j.models.chat.ChatModel;
import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder.ChatDemo;
import java.util.List;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.api.Test;

/**
 * Demo: mini GGUF — tokenize, forward, quality sampling (default).
 * Run via {@code mvn verify}.
 */
class MiniChatDemoIT {

    @Test
    @EnabledIf("com.github.tensor4j.models.chat.fixture.ChatDemoVocab#fullFixturePresent")
    void miniGgufTokenizesAndGeneratesQualityCompletions() throws Exception {
        ChatDemoReporter.banner("mini GGUF tokenize + quality generate");
        ChatModel model = ChatSession.loadOpenModel();
        ChatGenerationOptions options = ChatSession.optionsFor(model);
        ChatDemoReporter.modelInfo(model, "MiniChatGgufBuilder.buildOpenChatDemoModel()");

        assertTrue(model.config().nVocab() > 0);
        assertTrue(BpePreType.LLAMA3 == model.tokenizer().preType());
        assertTrue(options.mode() == ChatGenerationMode.QUALITY);
        assertTrue(model.config().nEmbd() == ChatDemo.N_EMBD);
        assertTrue(model.config().nLayer() == ChatDemo.N_LAYER);
        assertTrue(model.config().nCtx() == ChatDemo.N_CTX);
        assertTrue(model.config().nVocab() == ChatDemo.FULL_VOCAB);

        List<String> prompts = DemoPrompts.load();
        int ok = 0;
        for (String prompt : prompts) {
            if ("exit".equalsIgnoreCase(prompt)) {
                continue;
            }
            int[] promptIds = model.tokenizer().encode(prompt);
            ChatDemoReporter.tokenize(model.tokenizer(), prompt);
            if (promptIds.length == 0) {
                ChatDemoReporter.skipped(prompt, "no vocab match");
                continue;
            }
            float[] logits = model.forward(promptIds);
            assertFalse(Float.isNaN(logits[0]));
            assertTrue(logits.length == model.config().nVocab());

            ChatSession.GenerationResult generated = ChatSession.generate(model, prompt, options);
            ChatDemoReporter.generation(
                    prompt, generated.text(), generated.tokenCount(), generated.mode(), generated.prefixReuseTokens());
            if (!generated.text().isBlank()) {
                ChatSession.assertRealCompletion(generated.text());
                ok++;
            }
        }
        assertTrue(ok >= 1, "expected at least one non-empty completion");
        ChatDemoReporter.summary(prompts.size() - 1, ok);
    }
}
