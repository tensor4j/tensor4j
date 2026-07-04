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
import com.github.tensor4j.runtime.gguf.MmappedGgufFile;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

/**
 * Interactive stdin chat demo — only runs when {@code TENSOR4J_CHAT_INTERACTIVE=1}.
 * Session stays open up to 30 minutes; type {@code exit} to quit early.
 *
 * <p>Uses {@code TENSOR4J_GGUF_PATH} when set, otherwise the in-process mini fixture.
 */
@EnabledIfEnvironmentVariable(named = "TENSOR4J_CHAT_INTERACTIVE", matches = "1")
class InteractiveChatDemoIT {

    private static final long SESSION_MS = TimeUnit.MINUTES.toMillis(30);

    @Test
    @Timeout(value = 30, unit = TimeUnit.MINUTES)
    void interactiveReplSession() throws Exception {
        ChatDemoReporter.banner("interactive chat REPL (30 min max)");
        String ggufPath = System.getenv("TENSOR4J_GGUF_PATH");
        if (ggufPath != null && !ggufPath.isBlank()) {
            runWithExternal(Path.of(ggufPath));
            return;
        }
        ChatModel model = ChatSession.loadMiniModel();
        ChatDemoReporter.modelInfo(model, "mini fixture (set TENSOR4J_GGUF_PATH for real weights)");
        ChatRepl.run(model, System.currentTimeMillis() + SESSION_MS);
    }

    private static void runWithExternal(Path path) throws Exception {
        try (MmappedGgufFile mapped = MmappedGgufFile.open(path)) {
            ChatModel model = ChatModel.fromGguf(mapped);
            ChatDemoReporter.modelInfo(model, path.toAbsolutePath().toString());
            ChatRepl.run(model, System.currentTimeMillis() + SESSION_MS);
        }
    }
}
