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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.tensor4j.runtime.gguf.GgufHeader;
import com.github.tensor4j.runtime.gguf.MmappedGgufFile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

@EnabledIfEnvironmentVariable(named = "TENSOR4J_GGUF_PATH", matches = ".+")
class LiveQwenTokenizerLayoutTest {

    @Test
    void qwenEndTurnUsesImEndThenNewline() throws Exception {
        Path path = Paths.get(System.getenv("TENSOR4J_GGUF_PATH").trim());
        assertTrue(Files.isRegularFile(path));
        try (MmappedGgufFile mapped = MmappedGgufFile.open(path)) {
            GgufHeader header = mapped.header();
            ChatTokenizer tokenizer = ChatTokenizer.fromGguf(header);
            assertTrue(
                    tokenizer.preType() == BpePreType.QWEN2 || tokenizer.preType() == BpePreType.QWEN35,
                    tokenizer.preType().name());

            int[] endTurn = ChatTemplate.QWEN2.encodeEndTurn(tokenizer);
            assertTrue(endTurn.length >= 2, "Qwen end turn must be im_end + newline, got " + endTurn.length);
            assertEquals(tokenizer.endTurnId(), endTurn[0]);
            assertArrayEquals(tokenizer.encode("\n"), new int[] {endTurn[1]});

            int[] userTurn = ChatTemplate.QWEN2.encodeUserTurn(tokenizer, "hello");
            int[] prompt = ChatTemplate.QWEN2.encodePromptForGeneration(tokenizer, "hello");
            assertTrue(contains(userTurn, tokenizer.endTurnId()));
            assertTrue(prompt.length > userTurn.length);
        }
    }

    private static boolean contains(int[] array, int value) {
        for (int id : array) {
            if (id == value) {
                return true;
            }
        }
        return false;
    }
}
