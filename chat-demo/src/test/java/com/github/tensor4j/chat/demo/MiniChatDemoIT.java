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

import java.util.List;
import com.github.tensor4j.models.chat.BpePreType;
import com.github.tensor4j.models.chat.ChatGenerationMode;
import com.github.tensor4j.models.chat.ChatGenerationOptions;
import com.github.tensor4j.models.chat.ChatModel;
import org.junit.jupiter.api.Test;

/**
 * Demo: mini GGUF — tokenize, forward, quality sampling (default).
 * Run via {@code mvn verify}.
 */
class MiniChatDemoIT {

    @Test
    void miniGgufTokenizesAndGeneratesQualityCompletions() throws Exception {
        ChatDemoReporter.banner("mini GGUF tokenize + quality generate");
        ChatModel model = ChatSession.loadMiniModel();
        ChatGenerationOptions options = ChatSession.optionsFor(model);
        ChatDemoReporter.modelInfo(model, "MiniChatGgufBuilder.buildChatDemoModel()");

        assertTrue(model.config().nVocab() > 0);
        assertTrue(BpePreType.LLAMA3 == model.tokenizer().preType());
        assertTrue(options.mode() == ChatGenerationMode.QUALITY);

        List<String> prompts = DemoPrompts.load();
        int ok = 0;
        for (int i = 0; i < prompts.size(); i++) {
            String prompt = prompts.get(i);
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
            ChatDemoReporter.generation(prompt, generated.text(), generated.tokenCount(), generated.mode());
            if ("Hello".equals(prompt)) {
                assertFalse(generated.text().isBlank(), "Hello should produce a non-empty completion");
                assertTrue(generated.text().contains("there"), "Hello completion should include 'there'");
            }
            if (!generated.text().isBlank()) {
                ok++;
            }
        }
        assertTrue(ok >= 1, "expected at least one non-empty completion");
        ChatDemoReporter.summary(prompts.size() - 1, ok);
    }
}
