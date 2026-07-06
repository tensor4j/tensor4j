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

import com.github.tensor4j.models.chat.ChatMessage;
import com.github.tensor4j.models.chat.ChatTemplate;
import com.github.tensor4j.models.chat.ChatTokenizer;
import com.github.tensor4j.models.chat.LlamaCppChatApplier;
import com.github.tensor4j.runtime.gguf.MmappedGgufFile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

/** Dumps tensor4j ChatML token ids for llama.cpp parity checks (run with TENSOR4J_GGUF_PATH set). */
public final class LiveQwenChatTokenProbe {

    private static final String USER1 =
            "Write a simple Java program that prints \"Hello World\" to the console.";
    private static final String ASSISTANT1 =
            "Here is a simple Java program:\n\n```java\npublic class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello World\");\n    }\n}\n```";
    private static final String USER2 = "What is the difference between Spring and Java EE?";

    private LiveQwenChatTokenProbe() {}

    public static void main(String[] args) throws Exception {
        String ggufEnv = System.getenv("TENSOR4J_GGUF_PATH");
        if (ggufEnv == null || ggufEnv.isBlank()) {
            System.err.println("Set TENSOR4J_GGUF_PATH to a Qwen GGUF file.");
            System.exit(1);
        }
        Path ggufPath = Paths.get(ggufEnv.trim());
        if (!Files.isRegularFile(ggufPath)) {
            System.err.println("Missing GGUF: " + ggufPath);
            System.exit(1);
        }

        try (MmappedGgufFile mapped = MmappedGgufFile.open(ggufPath)) {
            ChatTokenizer tokenizer = ChatTokenizer.fromGguf(mapped.header());
            ChatTemplate template = ChatTemplate.fromTokenizer(tokenizer);
            LlamaCppChatApplier applier = LlamaCppChatApplier.fromTokenizer(tokenizer);

            List<ChatMessage> turn1 = List.of(new ChatMessage("user", USER1));
            int[] turn1Prefill = applier.tokenIds(tokenizer, turn1, true);
            System.out.println("tensor4j_turn1_prefill=" + Arrays.toString(turn1Prefill));

            List<ChatMessage> closed = List.of(new ChatMessage("user", USER1), new ChatMessage("assistant", ASSISTANT1));
            int[] turn1Closed = applier.tokenIds(tokenizer, closed, false);
            int prevTokens = applier.tokenCountAfterAssistantTurn(tokenizer, closed);
            System.out.println("tensor4j_turn1_closed=" + Arrays.toString(turn1Closed));
            System.out.println("tensor4j_prev_token_count=" + prevTokens);

            List<ChatMessage> turn2 =
                    List.of(new ChatMessage("user", USER1), new ChatMessage("assistant", ASSISTANT1), new ChatMessage("user", USER2));
            int[] turn2Full = applier.tokenIds(tokenizer, turn2, true);
            int[] turn2Delta = applier.tokenDeltaSince(tokenizer, turn2, true, prevTokens);
            System.out.println("tensor4j_turn2_delta=" + Arrays.toString(turn2Delta));
            System.out.println("tensor4j_turn2_full=" + Arrays.toString(turn2Full));

            int prevChars = applier.lengthAfterAssistantTurn(closed);
            String charDelta = applier.deltaSince(turn2, true, prevChars);
            int[] charDeltaIds = tokenizer.tokenizePrompt(charDelta, false);
            System.out.println("tensor4j_turn2_char_delta=" + Arrays.toString(charDeltaIds));
            System.out.println("tensor4j_prev_char_len=" + prevChars);

            int[] legacyTurn1 = template.encodePromptForGeneration(tokenizer, USER1);
            System.out.println("tensor4j_legacy_turn1=" + Arrays.toString(legacyTurn1));
            System.out.println(
                    "tensor4j_default_system="
                            + ChatTemplate.defaultSystemTurnEnabled());
            System.out.println(
                    "tensor4j_legacy_equals_applier="
                            + Arrays.equals(legacyTurn1, turn1Prefill));
        }
    }
}
