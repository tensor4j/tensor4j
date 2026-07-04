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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class TinygradRuntimeGoldenLoaderTest {

    @Test
    void loadsNineRuntimeCapturesWithOneDimensionalLogits() {
        var cases = TinygradRuntimeGoldenLoader.load();
        assertEquals(9, cases.size());
        for (TinygradRuntimeGoldenCase golden : cases) {
            assertEquals(1, golden.logitsShape().length, golden.name());
            assertEquals(golden.logitsShape()[0], golden.logits().length, golden.name());
            assertTrue(golden.runtimeToken() >= 0);
            assertTrue(golden.runtimeToken() < golden.vocabSize(), golden.name());
        }
    }

    @Test
    void gumbelAndLlamaPathsAreTagged() {
        long gumbel = TinygradRuntimeGoldenLoader.load().stream().filter(c -> "apps_llm_gumbel".equals(c.path())).count();
        long llama = TinygradRuntimeGoldenLoader.load().stream().filter(c -> "llama_sample".equals(c.path())).count();
        assertEquals(3, gumbel);
        assertEquals(6, llama);
    }
}
