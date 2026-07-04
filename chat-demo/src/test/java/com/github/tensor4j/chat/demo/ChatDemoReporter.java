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

import java.util.Locale;
import com.github.tensor4j.models.chat.ChatModel;
import com.github.tensor4j.models.chat.ChatTokenizer;

/** Prints a readable chat demo transcript to the Failsafe test log. */
final class ChatDemoReporter {

    private ChatDemoReporter() {
    }

    static void banner(String title) {
        System.out.println();
        System.out.println("=== tensor4j chat-demo: " + title + " ===");
    }

    static void modelInfo(ChatModel model, String source) {
        System.out.printf(
                Locale.US,
                "  model:     %s%n  layers:    %d   vocab: %d   ctx: %d   embd: %d%n",
                source,
                model.config().nLayer(),
                model.config().nVocab(),
                model.config().nCtx(),
                model.config().nEmbd());
    }

    static void tokenize(ChatTokenizer tokenizer, String prompt) {
        int[] ids = tokenizer.encode(prompt);
        System.out.printf(Locale.US, "  prompt:    %s%n", prompt);
        System.out.print("  tokens:   [");
        for (int i = 0; i < ids.length; i++) {
            if (i > 0) {
                System.out.print(", ");
            }
            System.out.print(ids[i]);
        }
        System.out.println("]");
    }

    static void skipped(String prompt, String reason) {
        System.out.printf(Locale.US, "  prompt:    %s%n  skipped:   %s%n", prompt, reason);
    }

    static void generation(String prompt, String completion, int tokensGenerated, String mode) {
        System.out.printf(java.util.Locale.US, "  prompt:    %s%n", prompt);
        System.out.printf(java.util.Locale.US, "  mode:      %s%n", mode);
        System.out.printf(java.util.Locale.US, "  completion:%s%n", completion.isEmpty() ? " (empty)" : " " + completion);
        System.out.printf(java.util.Locale.US, "  new tokens:%d%n", tokensGenerated);
    }

    static void summary(int prompts, int ok) {
        System.out.printf(Locale.US, "=== chat-demo pass: %d/%d prompts ===%n%n", ok, prompts);
    }
}
