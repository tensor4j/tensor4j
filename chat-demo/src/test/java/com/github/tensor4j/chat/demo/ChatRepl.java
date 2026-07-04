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

import com.github.tensor4j.models.chat.ChatModel;
import com.github.tensor4j.models.chat.ChatGenerationOptions;

/** Blocking stdin REPL for manual chat-demo sessions (up to 30 minutes). */
final class ChatRepl {

    private static final String EXIT = "exit";

    private ChatRepl() {
    }

    static void run(ChatModel model, long deadlineMs) throws java.io.IOException {
        ChatGenerationOptions options = ChatSession.optionsFor(model);
        System.out.println("Interactive chat — mode=" + options.mode()
                + " (TENSOR4J_CHAT_MODE=greedy for argmax). Type 'exit' to quit (30 minute limit).");
        java.io.BufferedReader reader = new java.io.BufferedReader(
                new java.io.InputStreamReader(System.in, java.nio.charset.StandardCharsets.UTF_8));
        while (System.currentTimeMillis() < deadlineMs) {
            System.out.print("you> ");
            System.out.flush();
            String line = reader.readLine();
            if (line == null) {
                break;
            }
            line = line.trim();
            if (line.isEmpty()) {
                continue;
            }
            if (EXIT.equalsIgnoreCase(line)) {
                System.out.println("chat-demo: goodbye.");
                break;
            }
            ChatSession.GenerationResult result = ChatSession.generate(model, line, options);
            System.out.printf(java.util.Locale.US, "bot> %s%n",
                    result.text().isEmpty() ? "(no tokens)" : result.text());
        }
        if (System.currentTimeMillis() >= deadlineMs) {
            System.out.println("chat-demo: 30 minute limit reached.");
        }
    }
}
