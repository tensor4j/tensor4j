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

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.github.tensor4j.models.chat.reference.TinygradGumbelSampleReference;
import com.github.tensor4j.models.chat.reference.TinygradLlamaSampleReference;
import com.github.tensor4j.models.chat.reference.TinygradRuntimeGoldenCase;
import com.github.tensor4j.models.chat.reference.TinygradRuntimeGoldenLoader;
import com.github.tensor4j.models.chat.reference.TinygradSamplingGoldenCases;
import org.junit.jupiter.api.Test;

/**
 * Parity against checked-in tinygrad Tensor runtime captures ({@code tinygrad-runtime-sampling-golden.json}).
 *
 * <p>No Python/tinygrad at test time — regenerate with {@code scripts/capture_tinygrad_runtime_golden.py}.
 */
class TinygradRuntimeGoldenParityTest {

    @Test
    void gumbelJavaReferenceMatchesTensorRuntime() {
        for (TinygradRuntimeGoldenCase runtime : TinygradRuntimeGoldenLoader.load()) {
            if (!"apps_llm_gumbel".equals(runtime.path())) {
                continue;
            }
            int javaToken = TinygradGumbelSampleReference.sample(
                    runtime.logits(), runtime.temperature(), runtime.tensorSeed());
            assertEquals(runtime.runtimeToken(), javaToken, runtime.name() + " java gumbel ref");
        }
    }

    @Test
    void gumbelChatSamplerMatchesTensorRuntime() {
        for (TinygradRuntimeGoldenCase runtime : TinygradRuntimeGoldenLoader.load()) {
            if (!"apps_llm_gumbel".equals(runtime.path())) {
                continue;
            }
            var golden = findPortGolden(runtime.name());
            assertEquals(runtime.runtimeToken(), TinygradSamplingGoldenCases.sampleWithChatSampler(golden));
        }
    }

    @Test
    void gumbelRuntimeMatchesPortGoldenJson() {
        for (TinygradRuntimeGoldenCase runtime : TinygradRuntimeGoldenLoader.load()) {
            if (!"apps_llm_gumbel".equals(runtime.path())) {
                continue;
            }
            var golden = findPortGolden(runtime.name());
            assertEquals(runtime.runtimeToken(), golden.expectedToken(), runtime.name());
        }
    }

    @Test
    void llamaAlphaTensorRuntimeMatchesJavaReferenceWithDocumentedRoll() {
        TinygradRuntimeGoldenCase runtime = TinygradRuntimeGoldenLoader.require("llama_alpha_fp_seed_0");
        TinygradLlamaSampleReference ref = new TinygradLlamaSampleReference(runtime.vocabSize());
        ref.record(1);
        ref.record(1);
        int javaToken = ref.sample(
                runtime.logits(),
                runtime.temperature(),
                0,
                1f,
                runtime.alphaFrequency(),
                runtime.alphaPresence(),
                0.5);
        assertEquals(runtime.runtimeToken(), javaToken);
        assertEquals(2, runtime.runtimeToken());
    }

    @Test
    void llamaTopKTensorRuntimeSeedZeroMatchesJavaReferenceWithDocumentedRoll() {
        TinygradRuntimeGoldenCase runtime = TinygradRuntimeGoldenLoader.require("llama_v5_topk2_topp09_seed_0");
        int javaToken = new TinygradLlamaSampleReference(runtime.vocabSize())
                .sample(runtime.logits(), runtime.temperature(), 2, 0.9f, 0f, 0f, 0.99);
        assertEquals(runtime.runtimeToken(), javaToken);
        assertEquals(1, runtime.runtimeToken());
    }

    @Test
    void llamaTensorMultinomialCapturesAreStableFixtures() {
        assertEquals(3, TinygradRuntimeGoldenLoader.require("llama_v4_tensor_seed_0").runtimeToken());
        assertEquals(3, TinygradRuntimeGoldenLoader.require("llama_v4_tensor_seed_99").runtimeToken());
        assertEquals(2, TinygradRuntimeGoldenLoader.require("llama_v5_topk2_topp09_seed_42").runtimeToken());
        assertEquals(1, TinygradRuntimeGoldenLoader.require("llama_alpha_fp_seed_17").runtimeToken());
    }

    private static com.github.tensor4j.models.chat.reference.TinygradSamplingGoldenCase findPortGolden(String name) {
        for (var golden : TinygradSamplingGoldenCases.all()) {
            if (golden.name().equals(name)) {
                return golden;
            }
        }
        throw new IllegalArgumentException("port golden missing: " + name);
    }
}
