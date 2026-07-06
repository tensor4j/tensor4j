/*
 * Copyright 2026 Tensor4j Maintainers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package com.github.tensor4j.gguf.it;

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.models.chat.ChatGenerationOptions;
import com.github.tensor4j.models.chat.ChatGenerationResult;
import com.github.tensor4j.models.chat.ChatGenerationStopReason;
import com.github.tensor4j.models.chat.ChatGenerator;
import com.github.tensor4j.models.chat.ChatHistoryMode;
import com.github.tensor4j.models.chat.ChatModel;
import com.github.tensor4j.models.chat.ChatTemplate;
import com.github.tensor4j.runtime.gguf.MmappedGgufFile;
import java.nio.file.Paths;
import java.util.Locale;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

/**
 * Real GGUF two-turn probe — run from {@code tensor4j-gguf-it/} after {@code mvn install} in parent.
 *
 * <p>Set {@code TENSOR4J_GGUF_PATH} to a local .gguf file. Heap: {@code failsafe.argLine} in this module's
 * {@code pom.xml} (default 10GB).
 */
@EnabledIfEnvironmentVariable(named = "TENSOR4J_GGUF_PATH", matches = ".+")
class ChatRealGgufDefectProbeIT {

    private static final String TURN1_USER = "What is 2+2?";
    private static final String TURN2_USER = "What is 7+8?";

    @Test
    void twoTurnArithmeticGreedyStopsOnEotNotMaxTokens() throws Exception {
        String path = System.getenv("TENSOR4J_GGUF_PATH").trim();
        try (MmappedGgufFile mapped = MmappedGgufFile.open(Paths.get(path))) {
            ChatModel model = ChatModel.fromGguf(mapped);
            runProbe(model);
        }
    }

    private static void runProbe(ChatModel model) {
        ChatTemplate template = ChatTemplate.fromTokenizer(model.tokenizer());
        ChatGenerationOptions options = ChatGenerationOptions.greedy(model.tokenizer(), 32);
        ChatGenerator generator = new ChatGenerator(model, options, ChatHistoryMode.LLAMA);

        ChatGenerationResult turn1 = generator.continueConversation(TURN1_USER, template);
        System.err.println("turn1 stop=" + turn1.stopReason() + " text=" + quote(turn1.text()));
        assertNotEquals(
                ChatGenerationStopReason.MAX_TOKENS,
                turn1.stopReason(),
                () -> "turn1 ran to max tokens — likely EOG/min_new spin");
        assertTrue(
                mentionsFour(turn1.text()),
                () -> "turn1 should answer 2+2 (substring 4 or word four): " + turn1.text());

        int[] turn2Plan = generator.planPromptIds(TURN2_USER, template);
        String turn2Prompt = model.tokenizer().decode(turn2Plan);
        System.err.println("turn2 plan tail=" + quote(tail(turn2Prompt, 280)));
        assertTrue(turn2Prompt.contains(TURN1_USER), () -> "turn-2 prompt must retain turn-1 user question: " + turn2Prompt);
        assertTrue(turn2Prompt.contains(TURN2_USER), () -> "turn-2 prompt missing turn-2 user question: " + turn2Prompt);
        assertTrue(
                turn2Prompt.indexOf(TURN2_USER) > turn2Prompt.indexOf(TURN1_USER),
                () -> "turn-2 question must follow turn-1 in prompt: " + turn2Prompt);

        ChatGenerationResult turn2 = generator.continueConversation(TURN2_USER, template);
        System.err.println("turn2 stop=" + turn2.stopReason() + " text=" + quote(turn2.text()));
        assertNotEquals(
                ChatGenerationStopReason.MAX_TOKENS,
                turn2.stopReason(),
                () -> "turn2 ran to max tokens");
        assertTrue(
                mentionsFifteen(turn2.text()),
                () -> "turn2 should answer 7+8 (substring 15 or word fifteen): " + turn2.text());
    }

    /** Fuzzy check: {@code 4} as substring or whole word {@code four}. */
    static boolean mentionsFour(String text) {
        String lower = text.toLowerCase(Locale.ROOT);
        return lower.contains("four") || lower.contains("4");
    }

    /** Fuzzy check: {@code 15} as substring or whole word {@code fifteen} (not lone {@code 1}). */
    static boolean mentionsFifteen(String text) {
        String lower = text.toLowerCase(Locale.ROOT);
        return lower.contains("fifteen") || lower.contains("15");
    }

    private static String quote(String s) {
        return "\"" + s.replace("\n", "\\n") + "\"";
    }

    private static String tail(String s, int max) {
        return s.length() <= max ? s : s.substring(s.length() - max);
    }
}
