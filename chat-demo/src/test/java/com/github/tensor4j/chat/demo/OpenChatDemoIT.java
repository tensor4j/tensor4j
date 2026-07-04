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
import com.github.tensor4j.models.chat.ChatTemplate;
import com.github.tensor4j.models.chat.fixture.MiniChatGgufBuilder.ChatDemo;
import java.util.List;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.api.Test;

/** Open-ended level-12 demo with optional llama3 template ({@code TENSOR4J_CHAT_TEMPLATE}). */
class OpenChatDemoIT {

    @Test
    @EnabledIf("com.github.tensor4j.models.chat.fixture.ChatDemoVocab#fullFixturePresent")
    void openModelTokenizesAndGeneratesRealCompletions() throws Exception {
        ChatDemoReporter.banner("open-ended mini GGUF generate");
        ChatModel model = ChatSession.loadOpenModel();
        ChatGenerationOptions options = ChatSession.optionsFor(model);
        ChatTemplate template = ChatSession.templateForDemo();
        ChatDemoReporter.modelInfo(model, "MiniChatGgufBuilder.buildOpenChatDemoModel()");

        assertTrue(BpePreType.LLAMA3 == model.tokenizer().preType());
        assertTrue(options.mode() == ChatGenerationMode.QUALITY);
        assertTrue(model.config().nEmbd() == ChatDemo.N_EMBD);
        assertTrue(model.config().nLayer() == ChatDemo.N_LAYER);
        assertTrue(model.config().nVocab() == ChatDemo.FULL_VOCAB);

        List<String> prompts = DemoPrompts.load();
        int ok = 0;
        for (String prompt : prompts) {
            if ("exit".equalsIgnoreCase(prompt)) {
                continue;
            }
            int[] promptIds = template.encodeUser(model.tokenizer(), prompt);
            ChatDemoReporter.tokenize(model.tokenizer(), prompt);
            if (promptIds.length == 0) {
                ChatDemoReporter.skipped(prompt, "no vocab match");
                continue;
            }
            float[] logits = model.forward(promptIds);
            assertFalse(Float.isNaN(logits[0]));
            assertTrue(logits.length == model.config().nVocab());

            ChatSession.GenerationResult generated = ChatSession.generate(model, prompt, options, template);
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
