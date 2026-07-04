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

/** One row from {@code tinygrad-runtime-sampling-golden.json} — tinygrad Tensor runtime capture. */
public record TinygradRuntimeGoldenCase(
        String name,
        String path,
        int[] logitsShape,
        float[] logits,
        float temperature,
        int tensorSeed,
        int topK,
        float topP,
        float alphaFrequency,
        float alphaPresence,
        int[] alphaCounts,
        int runtimeToken) {

    public int vocabSize() {
        return logitsShape[0];
    }
}
